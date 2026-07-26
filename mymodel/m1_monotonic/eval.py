"""M1 Phase 2 -- evaluation. Mirrors D1's timing metric EXACTLY
(mymodel/d1_align_matrix/eval.py: per-onset |t_pred - t_gt| via interpol_c2o,
same FPS/THRESHOLDS) so pct@0.5s is directly comparable to D1, B1a, CB_TA, and
the paper -- the only change is that the frame->position path comes from a
monotonic Viterbi decode over the onset-column alignment matrix
(extensions/alignment/monotonic_decode.py) instead of DTW over spatial columns.

Two decode-time-only fixes (no retraining -- both testable on an existing
checkpoint), applied after diagnosing the gap between M1's first real number
(63.3%) and B1a (88.9%):

1. SOFT POSITION READOUT (`soft_position_readout`). Ground truth in this
   project (confirmed: same `kind='previous'` convention as B1a/CB_TA) is a
   step function checked exactly AT onset instants, so "interpolating between
   onsets" was the wrong framing. The real issue: M1's Viterbi decode snaps
   HARD to one of N discrete onset columns, so a wrong pick costs a full
   inter-onset-gap error (confirmed: median inter-onset gap 0.174s, M1's
   median error was 0.300s -- same order of magnitude, consistent with
   "picked the adjacent onset" being the dominant error mode) -- whereas B1a's
   continuous per-pixel heatmap can be slightly imprecise without jumping a
   whole onset. Fix: blend onset_x over a LOCAL window of columns around the
   Viterbi pick, weighted by the alignment matrix's own softmax scores --
   graceful degradation instead of all-or-nothing, using machinery Phase 0
   already built (extensions/alignment/monotonic_decode.expected_position is
   the global analogue; this is the same idea windowed locally so it can't be
   pulled toward a distant, unrelated column).

2. ENTROPY-BASED repeat-ambiguity tagging (`_ambiguity_mask`), replacing the
   geometric `_repeat_mask` heuristic (which was clearly miscalibrated: 11/
   15632 onsets tagged on the first real run). Idea from Hentschel et al.,
   "Time to Align!" (TISMIR, transactions.ismir.net/articles/10.5334/
   tismir.296) -- NOT a score-following paper, a data-model paper for
   alignment provenance, but its central theme (alignment claims should carry
   explicit certainty, not just a position) maps directly onto something M1
   already produces for free: the alignment posterior at every frame. A
   repeat-ambiguous onset is exactly where that posterior is torn between two
   columns -- i.e. low top1-vs-top2 margin -- a self-diagnosing, principled
   signal from the model's own output instead of hand-tuned strip-x geometry.

    python -m mymodel.m1_monotonic.eval --config configs/m1_monotonic.yaml \
        --checkpoint results/m1_monotonic/best_model.pt --repeat_stratified
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


def soft_position_readout(S_np: np.ndarray, path: np.ndarray, onset_x: np.ndarray,
                          window: int = 2) -> np.ndarray:
    """S_np: (T, N) alignment scores (pre-softmax). path: (T,) Viterbi-decoded
    column per frame. onset_x: (N,) strip-x of each onset column. Returns (T,)
    soft x-position: at each frame, blend onset_x over columns within
    +/-window of the HARD decode, weighted by a LOCAL softmax of S_np at that
    frame -- local, not global, so it can only be pulled toward the decoded
    onset's immediate neighbors (adjacent-onset confusion), never toward a
    distant, unrelated column with some nonzero global softmax mass."""
    T, N = S_np.shape
    x_soft = np.empty(T, dtype=np.float64)
    for f in range(T):
        c = int(path[f])
        lo, hi = max(0, c - window), min(N - 1, c + window)
        local = S_np[f, lo:hi + 1]
        local = local - local.max()
        w = np.exp(local)
        w = w / w.sum()
        x_soft[f] = float((w * onset_x[lo:hi + 1]).sum())
    return x_soft


def _ambiguity_mask(S_np: np.ndarray, onset_frames: np.ndarray, frac: float) -> np.ndarray:
    """Entropy/margin-based repeat-ambiguity tag (see module docstring, idea
    2). At each onset frame, take the softmax posterior over ALL columns and
    compute the top1-vs-top2 probability margin -- small margin means the
    model's own alignment posterior is genuinely torn between two candidate
    columns, exactly the signature of a repeat (same-looking passage at two
    positions). Flags the bottom `frac` of onsets by margin (percentile-based,
    not an absolute threshold, since the raw score scale isn't independently
    calibrated) as ambiguous. Returns a bool mask, shape (len(onset_frames),).
    """
    N_onsets = len(onset_frames)
    margins = np.empty(N_onsets, dtype=np.float64)
    for i, f in enumerate(onset_frames):
        row = S_np[int(f)]
        p = np.exp(row - row.max())
        p = p / p.sum()
        top2 = np.partition(p, -2)[-2:]
        margins[i] = float(top2.max() - top2.min())
    thresh = np.quantile(margins, frac)
    return margins <= thresh


def eval_piece(model, piece, device, ambig_frac, soft_window):
    built = build_onset_columns(piece, t_max=None)
    if built is None:
        return None
    T, onset_frames, onset_cols, onset_x, gt_path = built
    with torch.no_grad():
        S = model(piece.mert[:T].to(device), piece.strip.to(device),
                  torch.from_numpy(onset_cols).to(device))          # (T, N)
        path, _ = viterbi_path(S)
    path = path.cpu().numpy()
    S_np = S.detach().cpu().numpy()
    add = piece.add_per_staff

    x_soft = soft_position_readout(S_np, path, onset_x, window=soft_window)
    ambig_mask = _ambiguity_mask(S_np, onset_frames, ambig_frac)

    diffs_hard, diffs_soft, is_ambig = [], [], []
    for i, f in enumerate(onset_frames):
        if f >= T:
            continue
        x_gt = onset_x[gt_path[f]] + add
        t_gt = float(piece.interpol_c2o(x_gt))

        x_pred_hard = onset_x[path[f]] + add
        t_pred_hard = float(piece.interpol_c2o(x_pred_hard))
        diffs_hard.append(abs(t_pred_hard - t_gt) / FPS)

        x_pred_soft = x_soft[f] + add
        t_pred_soft = float(piece.interpol_c2o(x_pred_soft))
        diffs_soft.append(abs(t_pred_soft - t_gt) / FPS)

        is_ambig.append(bool(ambig_mask[i]))
    return np.array(diffs_hard), np.array(diffs_soft), np.array(is_ambig)


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
    ap.add_argument('--ambig_frac', type=float, default=0.1,
                    help='fraction of onsets (by lowest posterior top1-top2 margin) tagged ambiguous')
    ap.add_argument('--soft_window', type=int, default=2,
                    help='+/- column window for the soft position readout blend')
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

    all_hard, all_soft, all_ambig = [], [], []
    for k, p in enumerate(pieces):
        r = eval_piece(model, p, device, a.ambig_frac, a.soft_window)
        if r is None:
            continue
        dh, ds, amb = r
        all_hard.append(dh); all_soft.append(ds); all_ambig.append(amb)
        if (k + 1) % 20 == 0:
            arr = np.concatenate(all_hard)
            print(f'  [{k+1}/{len(pieces)}] hard pct@0.5s so far = {100.0*(arr <= 0.5).mean():.1f}%', flush=True)

    hard = np.concatenate(all_hard); soft = np.concatenate(all_soft); amb = np.concatenate(all_ambig)
    print(f'\n=== M1 monotonic-Viterbi on MSMD {a.split} ({len(all_hard)} pieces) ===', flush=True)
    print('-- hard decode (snap to Viterbi-chosen onset column) --', flush=True)
    _report('ALL     ', hard)
    print('-- soft readout (blend +/- window columns by local posterior) --', flush=True)
    _report('ALL     ', soft)
    if a.repeat_stratified:
        print(f'\n-- entropy-tagged ambiguous onsets (bottom {a.ambig_frac*100:.0f}% by posterior margin) --',
              flush=True)
        _report('ambiguous   (hard)', hard[amb])
        _report('unambiguous (hard)', hard[~amb])
        _report('ambiguous   (soft)', soft[amb])
        _report('unambiguous (soft)', soft[~amb])
        print(f'  (ambiguous onsets: {amb.sum()}/{len(amb)} = {100.0*amb.mean():.1f}%)', flush=True)


if __name__ == '__main__':
    main()
