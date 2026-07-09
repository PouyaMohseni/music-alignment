"""D1 eval. For each test piece: build the (T, W_col) similarity matrix from the
trained towers, decode a frame->column path (offline DTW, or causal OLTW with
--online), convert each onset frame's path column back to strip-x pixels, and
feed the SAME interpol_c2o timing-error metric as eval_official -- so pct@0.5s is
directly comparable to CB_TA and every other model in this project.

    python -m mymodel.d1_align_matrix.eval --config configs/d1_align_matrix.yaml \
        --checkpoint results/d1_align_matrix/best_model.pt [--online]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mymodel.d1_align_matrix.model import D1Model
from mymodel.d1_align_matrix import data as d1data
from mymodel.d1_align_matrix.dtw import dtw_decode, oltw_decode

FPS = 20
THRESHOLDS = [0.05, 0.1, 0.5, 1.0, 5.0]


def eval_piece(model, piece, device, online=False, band_frac=0.05):
    """Returns list of per-onset timing errors (seconds)."""
    with torch.no_grad():
        S = model(piece.mert.to(device), piece.strip.to(device))   # (T, W_col)
    S_np = S.float().cpu().numpy()
    path_cols = oltw_decode(S_np) if online else dtw_decode(S_np, band_frac=band_frac)   # (T,)
    wd = piece.w_downsample
    diffs = []
    for f in piece.onset_frames:
        if f >= len(path_cols):
            continue
        x_pred = path_cols[f] * wd + wd / 2.0            # column -> strip-x px (center)
        # timing via interpol_c2o: map predicted strip-x and true strip-x to onset time
        x_pred_g = x_pred + piece.add_per_staff
        # true strip-x for this onset frame: invert via interpol_c2o at gt column.
        # We already have GT column per frame in piece.gt_cols.
        x_gt = float(piece.gt_cols[f].item()) * wd + wd / 2.0 + piece.add_per_staff
        t_pred = float(piece.interpol_c2o(x_pred_g))
        t_gt = float(piece.interpol_c2o(x_gt))
        diffs.append(abs(t_pred - t_gt) / FPS)
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d1_align_matrix.yaml')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--online', action='store_true', help='causal OLTW decode instead of offline DTW')
    ap.add_argument('--band_frac', type=float, default=0.05,
                    help='Sakoe-Chiba band as fraction of W for offline DTW (None-like: pass -1 to disable)')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    band_frac = None if a.band_frac is not None and a.band_frac < 0 else a.band_frac

    cfg = yaml.safe_load(open(a.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dc = cfg['data']

    model = D1Model(**cfg['model']).to(device)
    ck = torch.load(a.checkpoint, map_location=device)
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model.eval()
    print(f'Loaded D1 checkpoint {a.checkpoint} '
          f'({sum(p.numel() for p in model.parameters()):,} params) on {device}', flush=True)

    pieces = d1data.load_split(a.split, dc['processed_root'], dc['cpjku_data'],
                               dc['mert_roots'], dc['scale_factor'],
                               cfg['model']['w_downsample'], limit=a.limit)
    decoder = 'OLTW (causal/online)' if a.online else f'DTW (offline, band_frac={band_frac})'
    print(f'Decoding with {decoder} over {len(pieces)} pieces', flush=True)

    all_diffs = []
    for k, piece in enumerate(pieces):
        d = eval_piece(model, piece, device, online=a.online, band_frac=band_frac)
        all_diffs.extend(d)
        if (k + 1) % 10 == 0:
            arr = np.array(all_diffs)
            print(f'  [{k+1}/{len(pieces)}] pct@0.5s so far = '
                  f'{100.0*(arr <= 0.5).mean():.1f}%', flush=True)

    arr = np.array(all_diffs)
    print(f'\n=== D1 ({decoder}) on MSMD {a.split} ({len(pieces)} pieces) ===', flush=True)
    for th in THRESHOLDS:
        print(f'  <= {th}s: {100.0*(arr <= th).mean():.1f}%', flush=True)
    print(f'  mean error:   {arr.mean():.3f}s', flush=True)
    print(f'  median error: {np.median(arr):.3f}s', flush=True)
    print(f'  total onsets: {len(arr)}', flush=True)


if __name__ == '__main__':
    main()
