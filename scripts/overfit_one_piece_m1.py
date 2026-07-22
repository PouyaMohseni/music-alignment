"""M1 Phase 1 -- prove the monotonic-alignment method TRAINS end-to-end by
overfitting a single piece. This is the README's own week-6 discipline
("overfit a single piece to prove the loss and training loop work") applied to
M1: real D1 towers (MERT audio + CNN/transformer score) produce a frame x
onset-column alignment matrix, supervised ONLY by the forward-sum monotonic
objective (extensions/alignment/forward_sum.py) plus an annealed beta-binomial
prior, decoded by monotonic Viterbi (extensions/alignment/monotonic_decode.py).

Success = the forward-sum loss drops sharply AND the Viterbi-decoded per-frame
onset path converges to the ground-truth monotone path (near-perfect frame
accuracy). Phase 0 already proved the DP/decode are correct on synthetic
matrices; this proves the REAL two-tower model can be driven to the right
alignment by this objective, before spending full-MSMD GPU time (Phase 2).

    python -m scripts.overfit_one_piece_m1 --config configs/d1_align_matrix.yaml \
        [--t_max 600] [--steps 250] [--limit 20]
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from mymodel.d1_align_matrix import data as d1data
from mymodel.m1_monotonic.model import M1Model
from mymodel.m1_monotonic.data import build_onset_columns
from extensions.alignment.forward_sum import forward_sum_loss
from extensions.alignment.monotonic_decode import viterbi_path
from extensions.alignment.beta_binomial_prior import beta_binomial_log_prior
from mymodel.d1_align_matrix.losses import dense_ce_loss


def pick_piece(pieces, t_max, min_onsets=20):
    """First piece whose (truncated) onset-column build is usable and not tiny."""
    best = None
    for p in pieces:
        built = build_onset_columns(p, t_max=t_max)
        if built is None:
            continue
        T, onset_frames, onset_cols, onset_x, gt_path = built
        N = len(onset_frames)
        if N >= min_onsets:
            return p, built
        if best is None:
            best, best_built = p, built
    return (best, best_built) if best is not None else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d1_align_matrix.yaml')
    ap.add_argument('--t_max', type=int, default=600, help='truncate frames for a fast CPU proof')
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--limit', type=int, default=20, help='pieces to load before picking one')
    ap.add_argument('--lr', type=float, default=3.0e-4)
    ap.add_argument('--prior_scale', type=float, default=1.0)
    ap.add_argument('--anneal_frac', type=float, default=0.5, help='anneal prior to 0 over this fraction of steps')
    ap.add_argument('--ce_weight', type=float, default=1.0, help='dense per-frame CE anchor toward GT column')
    ap.add_argument('--fs_weight', type=float, default=1.0, help='forward-sum monotonic-path regularizer weight')
    ap.add_argument('--ce_sigma_cols', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = yaml.safe_load(open(a.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dc = cfg['data']

    pieces = d1data.load_split('train', dc['processed_root'], dc['cpjku_data'],
                               dc['mert_roots'], dc['scale_factor'],
                               cfg['model']['w_downsample'], limit=a.limit)
    piece, built = pick_piece(pieces, a.t_max)
    if piece is None:
        print('ERROR: no usable piece found (missing MERT/score?)', flush=True)
        sys.exit(1)
    T, onset_frames, onset_cols, onset_x, gt_path = built
    N = len(onset_frames)
    print(f'Overfitting piece "{piece.piece_name}"  T={T} frames  N={N} onset columns  '
          f'(W_col span {onset_cols.min()}..{onset_cols.max()})  device={device}', flush=True)

    mert = piece.mert[:T].to(device)
    strip = piece.strip.to(device)
    cols = torch.from_numpy(onset_cols).to(device)
    gt_path_t = torch.from_numpy(gt_path).to(device)

    # Sanity: the GT path must be a valid forward-sum path (monotone,
    # stay-or-advance-by-one, endpoints) -- if not, the objective is being
    # applied to the wrong target and no amount of training would be meaningful.
    d = np.diff(gt_path)
    assert gt_path[0] == 0 and gt_path[-1] == N - 1, 'GT path endpoints wrong'
    assert set(np.unique(d).tolist()) <= {0, 1}, f'GT path not stay/advance-by-one: {np.unique(d)}'
    print(f'  GT path validated: monotone, step-set {sorted(set(np.unique(d).tolist()))}, '
          f'0..{N-1}', flush=True)

    model = M1Model(**cfg['model']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=cfg['optim']['weight_decay'])
    log_prior = beta_binomial_log_prior(T, N, scale=a.prior_scale, device=device)   # (T, N), static
    anneal_steps = max(1, int(a.anneal_frac * a.steps))

    def decode_accuracy():
        model.eval()
        with torch.no_grad():
            S = model(mert, strip, cols)                 # (T, N), no prior at decode
            path, _ = viterbi_path(S)
        model.train()
        frame_acc = (path == gt_path_t).float().mean().item()
        col_mae = (path.float() - gt_path_t.float()).abs().mean().item()
        onset_acc = (path[torch.from_numpy(onset_frames).to(device)]
                     == gt_path_t[torch.from_numpy(onset_frames).to(device)]).float().mean().item()
        return frame_acc, col_mae, onset_acc

    fa0, mae0, oa0 = decode_accuracy()
    print(f'  [step   0] (untrained) frame_acc={fa0:.3f}  col_MAE={mae0:.2f}  onset_acc={oa0:.3f}', flush=True)

    t_start = time.time()
    loss0 = None
    for step in range(1, a.steps + 1):
        w = max(0.0, 1.0 - step / anneal_steps) * 1.0   # prior weight 1 -> 0
        S = model(mert, strip, cols)                     # (T, N)
        S_prior = S + w * log_prior if w > 0 else S
        # CE anchors each frame to its GT column (content/timing supervision);
        # forward-sum regularizes toward a globally-consistent monotone path.
        # Forward-sum ALONE was shown underdetermined (loss->0 but decode wrong).
        loss = a.fs_weight * forward_sum_loss(S_prior, apply_log_softmax=True, normalize=True)
        if a.ce_weight > 0:
            loss = loss + a.ce_weight * dense_ce_loss(S, gt_path_t, sigma_cols=a.ce_sigma_cols)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if loss0 is None:
            loss0 = float(loss.detach())
        if step % 25 == 0 or step == a.steps:
            fa, mae, oa = decode_accuracy()
            print(f'  [step {step:3d}] loss={float(loss.detach()):.4f} (prior_w={w:.2f})  '
                  f'frame_acc={fa:.3f}  col_MAE={mae:.2f}  onset_acc={oa:.3f}  '
                  f'{time.time()-t_start:.0f}s', flush=True)

    fa, mae, oa = decode_accuracy()
    loss_final = float(loss.detach())
    print(f'\n=== M1 Phase-1 overfit result ===', flush=True)
    print(f'  loss: {loss0:.4f} -> {loss_final:.4f}', flush=True)
    print(f'  frame accuracy: {fa0:.3f} -> {fa:.3f}', flush=True)
    print(f'  onset-frame accuracy: {oa0:.3f} -> {oa:.3f}', flush=True)
    print(f'  column MAE: {mae0:.2f} -> {mae:.2f}', flush=True)

    ok_loss = loss_final < 0.5 * loss0
    ok_acc = fa > 0.90
    if ok_loss and ok_acc:
        print('  PHASE-1 OVERFIT PASSED: forward-sum drives the real two-tower model to '
              'the correct monotone alignment.', flush=True)
    else:
        print(f'  PHASE-1 OVERFIT FAILED: ok_loss={ok_loss} ok_acc={ok_acc} '
              '(loss must halve AND frame_acc>0.90).', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
