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
import yaml

from precompute_mert_zenodo import _load_model, encode_wav, resample_emb, MERT_FPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wav_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--split_file', required=True,
                   help='cpjku_fmt/split_test.yaml -- ground truth for which piece names '
                        'to encode. A filename heuristic (exclude "_<digits>" suffixed wavs, '
                        'assuming those are tempo variants) is NOT reliable: some real piece '
                        'names legitimately end in a digit (e.g. "bwv-1006a_5", '
                        '"Czerny_Op_821_No_004"), which that heuristic wrongly excluded.')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--mert_id', default='m-a-p/MERT-v1-95M')
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading MERT ({a.mert_id}) on {device}...', flush=True)
    model = _load_model(a.mert_id, device)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    piece_names = yaml.safe_load(open(a.split_file))['files']
    wav_files = sorted(Path(a.wav_dir) / f'{name}.wav' for name in piece_names)
    missing = [w for w in wav_files if not w.exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} split pieces have no wav: {missing[:5]}')
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
