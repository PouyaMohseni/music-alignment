"""A1 -- AMT posteriorgram bank on CYOLO's frame grid.

WHY THIS FEATURE, AND WHY IT IS NOT A SYMBOLIC INTERMEDIATE
-----------------------------------------------------------
Our two-microphone measurement (results/amt_bridge_eval-437566.log) found that
room reverberation costs a piano AMT model essentially nothing: onset F1 0.9116
on the room mic vs 0.9124 on a direct pickup of the SAME take, at 50 ms
tolerance. On those same recordings our trackers lose ~30 points. That is a
property of the transcriber's learned REPRESENTATION -- it was trained on real
piano recordings -- not of the discrete notes it emits.

So we take the representation and leave the notes behind. What is written here
is the continuous 88-band pitch posterior, exactly as the network produces it,
with nothing thresholded, peak-picked, or decoded into events. It is a feature
map -- a learned pitch-salience transform that happens to be room-invariant --
in the same sense that a CQT or a MERT embedding is a feature map. No note
list, no MIDI, no symbolic score representation enters the model. That
distinction is the whole point: the decomposed pipeline (D1) routes through
symbolic events and is therefore off-target; this does not.

WHAT IS STORED
--------------
concat([frame_output, reg_onset_output]) -> (T, 176).
  frame_output      sustained pitch activity: WHAT is sounding
  reg_onset_output  onset regression: what just STARTED
Both are kept because the evaluation metric is scored at onsets, so the onset
channel carries the signal the task is graded on, while the frame channel
carries the harmonic context that disambiguates position.

FRAME GRID
----------
The AMT model runs at 100 fps (config.frames_per_second). CYOLO indexes
performances by frame at SAMPLE_RATE/HOP_SIZE = 22050/1102 = 20.0091 fps, NOT
20. Resampling to a rounded 20 would drift ~1.6 frames over three minutes --
under the 10-frame tolerance, so it would not fail loudly, it would just quietly
cost accuracy on the longest pieces and be near-impossible to attribute later.
We resample to the exact frame COUNT CYOLO's LogSpectrogram would have produced
for the same signal, reusing the helpers from precompute_mert_cyolo so the two
banks are index-compatible and can be concatenated per frame.

Output: {out_dir}/{piece}.npy, float16 (T, 176).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from scripts.precompute_mert_cyolo import (CY_FPS, n_frames_for, resample_to_n)

AMT_FPS = 100.0
AMT_SR = 16000
N_PITCH = 88
OUT_DIM = 2 * N_PITCH


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wav_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--checkpoint', default='/scratch/pmohseni/amt_ckpts/kong_stock.pth')
    p.add_argument('--device', default=None)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    a = p.parse_args()

    import librosa
    import soundfile as sf
    import torch
    from piano_transcription_inference import PianoTranscription

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Block the library's auto-download from silently replacing our checkpoint
    # with whatever it fetches: we must know exactly which weights produced the
    # bank. Same guard as scripts/amt_transcribe_real.py.
    import piano_transcription_inference.inference as _inf
    _real_getsize = os.path.getsize

    def _guarded_getsize(path, _c=os.path.abspath(a.checkpoint), _r=_real_getsize):
        if os.path.abspath(path) == _c:
            return 10 ** 9        # convince it the file is already complete
        return _r(path)

    _inf.os.path.getsize = _guarded_getsize
    try:
        import numpy.core.multiarray as _nma
        torch.serialization.add_safe_globals([_nma._reconstruct])
    except Exception:
        pass
    _real_load = torch.load

    def _compat_load(*ar, **kw):
        kw.setdefault('weights_only', False)
        return _real_load(*ar, **kw)

    torch.load = _compat_load
    try:
        tr = PianoTranscription(device=device, checkpoint_path=a.checkpoint)
        # Assert the weights actually loaded. load_state_dict(strict=False)
        # would otherwise leave a randomly initialised net happily emitting
        # garbage posteriors that look like a feature bank.
        ck = _real_load(a.checkpoint, map_location='cpu', weights_only=False)['model']
        # The Note_pedal checkpoint nests two sub-models; flattening with their
        # prefixes is required or every key mismatches. Without this the guard
        # below reports matched=0/540 -- which is exactly what it did on the
        # probe (job 773544), correctly refusing to emit a bank.
        if len(ck) <= 4 and all(isinstance(v, dict) for v in ck.values()):
            ck = {f'{pre}.{k}': v for pre in ck for k, v in ck[pre].items()}
        model = tr.model.module if hasattr(tr.model, 'module') else tr.model
        msd = model.state_dict()
        matched = [k for k in msd if k in ck and msd[k].shape == ck[k].shape]
        print(f'weights: model={len(msd)} ckpt={len(ck)} matched={len(matched)}', flush=True)
        if len(matched) < 0.9 * len(msd):
            raise RuntimeError(f'only {len(matched)}/{len(msd)} tensors matched '
                               f'{a.checkpoint} -- refusing to emit a bank from '
                               f'partly-random weights')
    finally:
        torch.load = _real_load
        _inf.os.path.getsize = _real_getsize

    os.makedirs(a.out_dir, exist_ok=True)
    wavs = sorted(f for f in os.listdir(a.wav_dir) if f.endswith('.wav'))
    mine = wavs[a.shard::a.num_shards]
    print(f'shard {a.shard}/{a.num_shards}: {len(mine)} of {len(wavs)} wavs', flush=True)
    print(f'  AMT {AMT_FPS:.0f} fps -> CYOLO {CY_FPS:.6f} fps, out_dim={OUT_DIM}', flush=True)

    done = skip = fail = 0
    for i, name in enumerate(mine):
        key = os.path.splitext(name)[0]
        out_path = os.path.join(a.out_dir, key + '.npy')
        if os.path.exists(out_path):
            skip += 1
            continue
        wav_path = os.path.join(a.wav_dir, name)
        t0 = time.time()
        try:
            info = sf.info(wav_path)
            # frame count in CYOLO's own sample rate, which its indices assume
            n_samples = int(round(info.frames * 22050 / info.samplerate))
            n_dst = n_frames_for(n_samples)

            audio, _ = librosa.load(wav_path, sr=AMT_SR, mono=True)
            res = tr.transcribe(audio, None)
            od = res['output_dict']
            post = np.concatenate([np.asarray(od['frame_output'], dtype=np.float32),
                                   np.asarray(od['reg_onset_output'], dtype=np.float32)],
                                  axis=1)                       # (T_amt, 176)
            if post.shape[1] != OUT_DIM:
                raise ValueError(f'expected {OUT_DIM} dims, got {post.shape}')
            if post.shape[0] == 0:
                raise ValueError('empty posteriorgram')

            # Durations must agree. If they do not, resample_to_n's clipping
            # would flat-line the tail into a repeated final frame instead of
            # failing, and training would consume a constant feature for the
            # end of the piece.
            dur_src = post.shape[0] / AMT_FPS
            dur_dst = n_dst / CY_FPS
            if abs(dur_src - dur_dst) > max(0.5, 0.02 * dur_dst):
                raise ValueError(f'duration mismatch: AMT {dur_src:.2f}s vs '
                                 f'CYOLO grid {dur_dst:.2f}s ({n_dst} frames)')

            out = resample_to_n(post, AMT_FPS, n_dst)
            assert out.shape == (n_dst, OUT_DIM), f'{key}: {out.shape}'
            np.save(out_path, out.astype(np.float16))
            done += 1
        except Exception as e:
            print(f'  FAIL {key}: {type(e).__name__}: {e}', flush=True)
            fail += 1
        if (i + 1) % 10 == 0 or i + 1 == len(mine):
            print(f'  [{i+1}/{len(mine)}] done={done} skip={skip} fail={fail} '
                  f'({time.time()-t0:.1f}s last)', flush=True)

    print(f'Done shard {a.shard}. done={done} skip={skip} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
