"""M1 Phase 2 -- evaluation. Mirrors D1's timing metric EXACTLY
(mymodel/d1_align_matrix/eval.py: per-onset |t_pred - t_gt| via interpol_c2o,
same FPS/THRESHOLDS) so pct@0.5s is directly comparable to D1, B1a, CB_TA, and
the paper -- the only change is that the frame->position path comes from a
monotonic Viterbi decode over the onset-column alignment matrix
(extensions/alignment/monotonic_decode.py) instead of DTW over spatial columns.

Also reports a REPEAT-STRATIFIED breakdown (the headline M1 claim): pct@0.5s on
repeat-ambiguous onsets vs the rest. Repeat tagging here is a zero-extra-data
STRUCTURAL proxy -- an onset is repeat-ambiguous if the same score-x is visited
again at a far-apart frame (|dx| <= x_tol AND |dframe| >= frame_gap). This
deliberately avoids reconciling the pitch-based find_repeat_groups column space
against D1's downsampled-strip column space (a coordinate-mismatch class of bug
that has repeatedly bitten this project); it measures exactly the thing M1
targets -- one score position played at two very different times, which a
monotone global path can disambiguate and a per-frame classifier cannot. A
pitch-based cross-check is a later refinement (see M1.md).

    python -m mymodel.m1_monotonic.eval --config configs/m1_monotonic.yaml \
        --checkpoint results/m1_monotonic/best_model.pt [--repeat_stratified]
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mymodel.d1_align_matrix import data as d1data
from mymodel.m1_monotonic.model import M1Model
from mymodel.m1_monotonic.data import build_onset_columns
from extensions.alignment.monotonic_decode import viterbi_path

FPS = 20
THRESHOLDS = [0.05, 0.1, 0.5, 1.0, 5.0]


def _repeat_mask(onset_frames, onset_x, x_tol, frame_gap):
    """Structural repeat tag: onset i is ambiguous if some other onset j shares
    its score-x (within x_tol) at a far-apart frame (>= frame_gap)."""
    N = len(onset_frames)
    mask = np.zeros(N, dtype=bool)
    for i in range(N):
        dx = np.abs(onset_x - onset_x[i])
        dframe = np.abs(onset_frames - onset_frames[i])
        near_far = (dx <= x_tol) & (dframe >= frame_gap)
        near_far[i] = False
        if near_far.any():
            mask[i] = True
    return mask


def eval_piece(model, piece, device, x_tol, frame_gap):
    built = build_onset_columns(piece, t_max=None)
    if built is None:
        return None
    T, onset_frames, onset_cols, onset_x, gt_path = built
    with torch.no_grad():
        S = model(piece.mert[:T].to(device), piece.strip.to(device),
                  torch.from_numpy(onset_cols).to(device))          # (T, N)
        path, _ = viterbi_path(S)
    path = path.cpu().numpy()
    add = piece.add_per_staff

    diffs, is_rep = [], []
    rep_mask = _repeat_mask(onset_frames, onset_x, x_tol, frame_gap)
    # gt_path[f] at an onset frame f is that onset's own index; path[f] is the
    # decoded onset column. Map both columns -> strip-x -> time via interpol_c2o.
    onset_index = {int(f): i for i, f in enumerate(onset_frames)}
    for i, f in enumerate(onset_frames):
        if f >= T:
            continue
        x_pred = onset_x[path[f]] + add
        x_gt = onset_x[gt_path[f]] + add
        t_pred = float(piece.interpol_c2o(x_pred))
        t_gt = float(piece.interpol_c2o(x_gt))
        diffs.append(abs(t_pred - t_gt) / FPS)
        is_rep.append(bool(rep_mask[i]))
    return np.array(diffs), np.array(is_rep)


def _report(name, arr):
    if len(arr) == 0:
        print(f'  {name}: (no onsets)', flush=True); return
    line = '  '.join(f'<={th}s:{100.0*(arr <= th).mean():5.1f}%' for th in THRESHOLDS)
    print(f'  {name} (n={len(arr)}): {line}  mean={arr.mean():.3f}s med={np.median(arr):.3f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/m1_monotonic.yaml')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--repeat_stratified', action='store_true')
    ap.add_argument('--x_tol_cols', type=float, default=1.5,
                    help='repeat x-tolerance in onset-column units (converted to strip-x px)')
    ap.add_argument('--repeat_gap_sec', type=float, default=2.0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dc = cfg['data']; wd = cfg['model']['w_downsample']

    model = M1Model(**cfg['model']).to(device)
    ck = torch.load(a.checkpoint, map_location=device)
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model.eval()
    print(f'[M1 eval] loaded {a.checkpoint} ({sum(p.numel() for p in model.parameters()):,} params) '
          f'on {device}', flush=True)

    pieces = d1data.load_split(a.split, dc['processed_root'], dc['cpjku_data'], dc['mert_roots'],
                               dc['scale_factor'], wd, limit=a.limit)
    x_tol = a.x_tol_cols * wd            # onset_x is in /scale_factor strip px; wd px = one column
    frame_gap = a.repeat_gap_sec * FPS

    all_diffs, all_rep = [], []
    for k, p in enumerate(pieces):
        r = eval_piece(model, p, device, x_tol, frame_gap)
        if r is None:
            continue
        d, rep = r
        all_diffs.append(d); all_rep.append(rep)
        if (k + 1) % 20 == 0:
            arr = np.concatenate(all_diffs)
            print(f'  [{k+1}/{len(pieces)}] pct@0.5s so far = {100.0*(arr <= 0.5).mean():.1f}%', flush=True)

    arr = np.concatenate(all_diffs); rep = np.concatenate(all_rep)
    print(f'\n=== M1 monotonic-Viterbi on MSMD {a.split} ({len(all_diffs)} pieces) ===', flush=True)
    _report('ALL     ', arr)
    if a.repeat_stratified:
        _report('repeat  ', arr[rep])
        _report('nonrepeat', arr[~rep])
        print(f'  (repeat onsets: {rep.sum()}/{len(rep)} = {100.0*rep.mean():.1f}%)', flush=True)


if __name__ == '__main__':
    main()
