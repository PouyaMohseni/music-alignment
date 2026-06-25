"""Create per-split symlink directories for CPJKU native train_model.py.

train_model.py takes --train_set and --val_set as separate directory paths
and globs all *.npz from <dir>/score/.  Since we keep all pieces in one
data/MSMD/cpjku_fmt/, we create split-specific subdirectories with symlinks:

    data/MSMD/cpjku_fmt/train/score/<pid>.npz  → ../../../score/<pid>.npz
    data/MSMD/cpjku_fmt/train/performance/...  → symlinks to parent perf dir
    data/MSMD/cpjku_fmt/val/...
    data/MSMD/cpjku_fmt/test/...

    python -m mymodel.cpjku_adapter.split_cpjku_data \
        --processed data/MSMD/processed \
        --cpjku_fmt data/MSMD/cpjku_fmt
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def create_split_dir(cpjku_fmt: Path, split_name: str, piece_ids: list[str]):
    out       = cpjku_fmt / split_name
    score_dir = out / 'score';       score_dir.mkdir(parents=True, exist_ok=True)
    perf_dir  = out / 'performance'; perf_dir.mkdir(parents=True, exist_ok=True)

    src_score = cpjku_fmt / 'score'
    src_perf  = cpjku_fmt / 'performance'

    linked = skipped = 0
    for pid in piece_ids:
        npz_src = src_score / f'{pid}.npz'
        if not npz_src.exists():
            skipped += 1
            continue

        # Score NPZ
        dst = score_dir / f'{pid}.npz'
        if not dst.exists():
            dst.symlink_to(npz_src.resolve())

        # All performance files for this piece
        for fname in (f'{pid}.wav', f'{pid}_1000.wav', f'{pid}.mid'):
            src = src_perf / fname
            dst = perf_dir / fname
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())

        linked += 1

    print(f'  {split_name}: {linked} pieces linked, {skipped} skipped (not converted)')
    return linked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--cpjku_fmt', default='data/MSMD/cpjku_fmt')
    a = p.parse_args()

    proc   = Path(a.processed)
    cpjku  = Path(a.cpjku_fmt)
    splits = json.load(open(proc / 'splits.json'))

    print(f'Creating per-split directories under {cpjku}/')
    for split_name, piece_ids in splits.items():
        create_split_dir(cpjku, split_name, piece_ids)


if __name__ == '__main__':
    main()
