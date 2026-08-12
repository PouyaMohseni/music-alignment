"""H1 -- serve MERT embeddings to CYOLO in place of its mel spectrogram.

Three surgical changes, no third-party source edited:

  1. `load_dataset` stores a FLATTENED MERT array per piece instead of a
     waveform, with `hop_length = frame_size = D` (768).
  2. `YOLO.compute_spec` reshapes that flat slice back to (T, D) instead of
     running LogSpectrogram.
  3. `ContextConditioning.enc` becomes a MERTWindowEncoder
     (extensions/heads/mert_cyolo_projector.py).

WHY FLATTENED, WITH hop_length = D
----------------------------------
The obvious approach -- store (T, D) and set hop_length = 1 -- breaks on
dataset.py:119, which augments by

    truncated_signal = np.pad(truncated_signal, (n * self.hop_length, 0), 'constant')

On a 2-D array `np.pad(a, (n, 0))` pads BOTH axes, silently adding n rows AND n
columns of zeros to the feature dimension. Storing the sequence flat as
(T*D,) with hop_length = D makes every existing index computation correct
without touching a line of their code:

    start_t = start_frame * D
    t       = D + frame * D          = (frame + 1) * D
    slice   = flat[start_frame*D : (frame+1)*D]     -> frames [start_frame, frame]

and the pad above prepends exactly n whole zero-frames, which is what it means
to pad by n frames. The frame span matches CYOLO's spectrogram span exactly
because scripts/precompute_mert_cyolo.py built the bank against CYOLO's own
frame-count formula, 1 + (n_samples - FRAME_SIZE)//HOP_SIZE, at its true
20.0091 fps rather than a rounded 20.

IMPULSE-RESPONSE AUGMENTATION IS NOT AVAILABLE IN THIS MODE
-----------------------------------------------------------
CYOLO's ImpulseResponse transform convolves `sample['performance']`, which is a
waveform. Convolving a precomputed embedding sequence is meaningless, so
`--ir_path` MUST NOT be combined with this patch; the guard below raises rather
than silently producing garbage. Since IR augmentation is worth a great deal to
this architecture (our own reproduction: 67.1 on room at 18% of training, with
IR), a fair comparison against the IR-trained baseline needs a SECOND,
IR-degraded MERT bank served with some probability -- the same multi-condition
construction used for R2r_realir. Until that bank exists, the honest comparison
for this run is against the NO-IR baseline row, not against 79.9.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

MERT_DIM = 768


def _load_bank(emb_root: str) -> dict:
    files = sorted(Path(emb_root).glob('*.npy'))
    if not files:
        raise RuntimeError(f'MERT bank {emb_root!r} contains no .npy files')
    return {f.stem: str(f) for f in files}


def patch_cyolo_mert(emb_roots, spec_out: int = 32, strict: bool = True,
                     aug_roots=None, aug_prob: float = 0.0):
    """emb_roots: {dataset_dir: mert_emb_dir}. Call BEFORE load_dataset.

    aug_roots/aug_prob enable MULTI-CONDITION training: a deterministic
    per-piece fraction `aug_prob` is served from an IR-degraded bank instead of
    the clean one.  This is the only way to give this architecture the
    augmentation that is worth the most to it (our own reproduction: 67.1 on
    room at 18% of training WITH real IRs), because CYOLO's waveform-level
    ImpulseResponse transform cannot operate on precomputed features.

    Assignment is by hash of the piece name, not by RNG draw, so a piece is
    always served in the same condition across epochs and workers -- a
    per-batch coin flip would let the model see both versions of the same piece
    and learn to average them.
    """
    import hashlib

    import cyolo_score_following.dataset as ds_mod
    from cyolo_score_following.models import conditioning_networks as cond_mod
    from extensions.heads.mert_cyolo_projector import MERTWindowEncoder

    banks = {k: _load_bank(v) for k, v in emb_roots.items()}
    total = sum(len(b) for b in banks.values())
    print(f'[H1] MERT banks: ' +
          ', '.join(f'{k}->{len(v)} npy' for k, v in banks.items()), flush=True)

    aug_banks = {}
    if aug_roots and aug_prob > 0.0:
        aug_banks = {k: _load_bank(v) for k, v in aug_roots.items()}
        n_aug = sum(len(b) for b in aug_banks.values())
        print(f'[H1] multi-condition ACTIVE: p(degraded)={aug_prob}, '
              f'{n_aug} degraded npy', flush=True)
        # compare against the CLEAN count for the same datasets the degraded
        # map covers -- comparing train-only degraded against train+valid
        # clean fired a false 94.9% warning on job 770876.
        clean_same = sum(len(banks[k]) for k in aug_banks if k in banks)
        if clean_same and n_aug < 0.98 * clean_same:
            print(f'[H1] WARNING: degraded bank covers {100.0*n_aug/total:.1f}% of the '
                  f'clean bank; effective augmentation rate is below {aug_prob}', flush=True)
    elif aug_roots or aug_prob:
        raise RuntimeError('aug_roots and aug_prob must be set together')

    def _use_degraded(name: str) -> bool:
        if not aug_banks:
            return False
        h = hashlib.sha1(name.encode()).digest()
        return (int.from_bytes(h[:4], 'big') / 0xFFFFFFFF) < aug_prob

    def _lookup(piece_name):
        if _use_degraded(piece_name):
            for b in aug_banks.values():
                if piece_name in b:
                    return b[piece_name]
            # fall through to clean rather than dropping the piece, but this
            # dilutes the intervention, so it is surfaced in the count below.
        for b in banks.values():
            if piece_name in b:
                return b[piece_name]
        return None

    # ---- 1. dataset serves flattened MERT instead of waveforms --------------
    _orig_load_dataset = ds_mod.load_dataset

    def load_dataset(*a, **kw):
        if kw.get('ir_path') is not None or (len(a) > 4 and a[4] is not None):
            raise RuntimeError(
                '--ir_path cannot be combined with the MERT patch: CYOLO applies '
                'impulse responses to the WAVEFORM, and convolving a precomputed '
                'embedding sequence is meaningless. Build an IR-degraded MERT '
                'bank and serve it multi-condition instead.')
        dataset = _orig_load_dataset(*a, **kw)

        missing, ok = [], 0
        # piece_names is a DICT {i: name} (dataset.py:269,308), keyed identically
        # to performances. enumerate() over it yields the int KEYS, not the
        # names, which is what killed job 773292 ('int' has no attribute
        # 'encode'). .items() gives the pairing the lookup actually needs.
        for i, name in dataset.piece_names.items():
            p = _lookup(name)
            if p is None:
                missing.append(name)
                continue
            emb = np.load(p).astype(np.float32)          # (T, 768)
            if emb.shape[1] != MERT_DIM:
                raise RuntimeError(f'{name}: expected dim {MERT_DIM}, got {emb.shape}')
            flat = emb.reshape(-1)                       # (T*768,)
            # one signal per piece here, but tolerate a list rather than
            # silently storing an array where a list was expected
            cur = dataset.performances.get(i) if hasattr(dataset.performances, 'get') \
                else dataset.performances[i]
            dataset.performances[i] = [flat] * len(cur) if isinstance(cur, (list, tuple)) else flat
            ok += 1

        if missing:
            msg = (f'[H1] {len(missing)} of {len(dataset.piece_names)} pieces have no '
                   f'MERT embedding (e.g. {missing[:3]})')
            if strict:
                raise RuntimeError(msg + ' -- refusing to train on a partial bank, '
                                         'which would silently mix representations')
            print('WARNING ' + msg, flush=True)

        dataset.frame_size = MERT_DIM
        dataset.hop_length = MERT_DIM
        print(f'[H1] dataset serving MERT for {ok} pieces '
              f'(frame_size=hop_length={MERT_DIM})', flush=True)
        return dataset

    ds_mod.load_dataset = load_dataset

    # ---- 2. compute_spec becomes a reshape ---------------------------------
    # The class is `Model`, not `YOLO` (models/yolo.py:78) -- job 770876
    # died on that import after 4 minutes of A100 time.
    from cyolo_score_following.models.yolo import Model as _CyoloModel
    from extensions.heads.mert_cyolo_projector import resample_frames

    def compute_spec(self, x, tempo_aug=False):
        out = []
        for item in x:
            flat = item if torch.is_tensor(item) else torch.as_tensor(item)
            n = flat.numel() // MERT_DIM
            emb = flat[:n * MERT_DIM].reshape(n, MERT_DIM)
            if tempo_aug:
                # CYOLO tempo-augments with a phase vocoder on the waveform,
                # impossible on precomputed features. Frame-axis resampling
                # reproduces what the augmentation is FOR (invariance to how
                # fast the score is traversed) but not its acoustics.
                factor = float(np.random.uniform(0.5, 2.0))
                emb = resample_frames(emb, factor)
            out.append(emb)
        return out

    _CyoloModel.compute_spec = compute_spec

    # ---- 3. swap the window encoder ----------------------------------------
    _orig_cc_init = cond_mod.ContextConditioning.__init__

    def cc_init(self, *a, **kw):
        _orig_cc_init(self, *a, **kw)
        so = kw.get('spec_out', spec_out)
        self.enc = MERTWindowEncoder(in_dim=MERT_DIM, spec_out=so, kw=self.kw,
                                     groupnorm=kw.get('groupnorm', False))
        self.kh = MERT_DIM
        print(f'[H1] ContextConditioning.enc -> MERTWindowEncoder '
              f'(in={MERT_DIM}, out={so}, kw={self.kw})', flush=True)

    cond_mod.ContextConditioning.__init__ = cc_init

    print('[H1] MERT-in-CYOLO patch ACTIVE', flush=True)
    return total
