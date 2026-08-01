"""Re-encode MERT embeddings from an ACOUSTIC-SHIFT tier's wavs.

WITHOUT THIS, EVERY MERT RESULT ON THE TIER IS A LIE. extensions/hooks/
mert_patch.py does not encode audio at eval time -- `_load_mert_spec` resolves
a piece to a precomputed file, `{emb_root}/{piece}_tempo_{T}.npy`. Those
existing embeddings were encoded from the ORIGINAL clean MSMD renders. Point a
MERT checkpoint at a reverberant, differently-synthesised tier without redoing
this step and it still reads the CLEAN features: the degradation never reaches
the model, the score is unchanged, and the run appears to show that MERT is
perfectly robust to acoustic shift. That conclusion would be an artefact of
the caching, not a property of MERT.

So the tier's audio must be pushed through MERT itself, which is also the
actual scientific question: MERT is pretrained on real recordings, so it may
close the synthetic->real gap in a way a mel-spectrogram CNN trained only on
one soundfont cannot.

NAMING. The tier stores audio as `{piece}_{tempo}.wav` (the layout
real_perf=True expects), while mert_patch looks up `{piece}_tempo_{tempo}.npy`.
This script performs exactly that rename so the existing patch works unchanged.

    python -m scripts.precompute_mert_acoustic_tier \
        --tier_dir /scratch/pmohseni/acoustic_tiers/room \
        --out_dir  /scratch/pmohseni/mert_emb_acoustic/room
    # then: MERT_TEST_EMB_ROOT=/scratch/pmohseni/mert_emb_acoustic/room
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

from precompute_mert_zenodo import _load_model, encode_wav, resample_emb, MERT_FPS   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tier_dir', required=True, help='acoustic tier root (has performance/*.wav)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--tempo', type=int, default=1000)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--mert_id', default='m-a-p/MERT-v1-95M')
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args()

    perf = Path(a.tier_dir) / 'performance'
    wavs = sorted(perf.glob(f'*_{a.tempo}.wav'))
    if a.limit:
        wavs = wavs[:a.limit]
    if not wavs:
        raise SystemExit(f'no *_{a.tempo}.wav under {perf}')

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'encoding {len(wavs)} wavs from {perf}', flush=True)
    print(f'  device={device}  model={a.mert_id}  -> {out}', flush=True)
    model = _load_model(a.mert_id, device)

    suffix = f'_{a.tempo}'
    done = 0
    for i, w in enumerate(wavs, 1):
        piece = w.stem[:-len(suffix)] if w.stem.endswith(suffix) else w.stem
        # mert_patch looks this exact key up; do not change it.
        dst = out / f'{piece}_tempo_{a.tempo}.npy'
        if dst.exists():
            done += 1
            continue
        # Same calls precompute_mert_zenodo makes, so the tier embeddings are
        # produced identically to the training/eval ones -- only the audio
        # differs, which is the whole point of the tier.
        emb = encode_wav(model, str(w), device=str(device))   # (T, 768) @ MERT_FPS
        emb = resample_emb(emb, MERT_FPS, a.fps)               # (T', 768) @ fps
        np.save(dst, emb.astype(np.float32))
        done += 1
        if i % 10 == 0 or i == len(wavs):
            print(f'  [{i}/{len(wavs)}] {piece}  {emb.shape}', flush=True)

    print(f'\ndone: {done} embeddings -> {out}', flush=True)
    print(f'set MERT_TEST_EMB_ROOT={out}', flush=True)


if __name__ == '__main__':
    main()
