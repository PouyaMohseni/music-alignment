"""Precompute MERT-v1-95M embeddings for B1a's eval, keyed by whole-piece
test wavs -- NOT the same files precompute_mert_zenodo.py used for training.

Training used per-page MIDI renders (data/msmd_test/performance/{piece}_page_N_tempo_T.mid,
matching train_model.py's own load_piece which trains on one page at a time).
But mymodel/cpjku_adapter/eval_official.py evaluates from a SINGLE concatenated
wav per whole piece (data/MSMD/cpjku_fmt/performance/{piece}.wav, built by
convert.py from the piece's full, unpaginated MIDI) -- a completely different
set of audio files with no page split. B1a's eval was pointed at the training
embeddings and got FileNotFoundError (no page-less key exists there); this
script fills that actual gap.

    python scripts/precompute_mert_test_eval.py \
        --wav_dir  data/MSMD/cpjku_fmt/performance \
        --out_dir  /scratch/pmohseni/mert_emb_zenodo/cpjku_fmt_test_eval
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from precompute_mert_zenodo import _load_model, encode_wav, resample_emb, MERT_FPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wav_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--mert_id', default='m-a-p/MERT-v1-95M')
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading MERT ({a.mert_id}) on {device}...', flush=True)
    model = _load_model(a.mert_id, device)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only whole-piece wavs, e.g. "AndreJ__O34__andre-sonatine.wav" -- exclude
    # the "_1000"-suffixed tempo-scaled wavs also present in this directory.
    wav_files = sorted(w for w in Path(a.wav_dir).glob('*.wav') if not w.stem.split('_')[-1].isdigit())
    print(f'{len(wav_files)} whole-piece wavs to encode', flush=True)

    done = skip = fail = 0
    for i, wav_path in enumerate(wav_files):
        key = wav_path.stem
        out_path = out_dir / f'{key}.npy'
        if out_path.exists():
            skip += 1
            continue
        try:
            emb = encode_wav(model, wav_path, device=device)
            if emb.shape[0] == 0:
                print(f'  SKIP {key}: empty audio', flush=True)
                fail += 1
                continue
            emb20 = resample_emb(emb, MERT_FPS, a.fps)
            np.save(out_path, emb20.astype(np.float16))
            done += 1
        except Exception as e:
            print(f'  FAIL {key}: {e}', flush=True)
            fail += 1
        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(wav_files)}] done={done} skip={skip} fail={fail}', flush=True)

    print(f'Finished: done={done} skip={skip} fail={fail} total={len(wav_files)}', flush=True)


if __name__ == '__main__':
    main()
