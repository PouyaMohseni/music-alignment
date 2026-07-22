"""Decisive test: is frozen MuSViT (+ trainable projection/context adapter)
usable as M1's score tower? Same overfit-one-piece bar Phase 1 already
validated for the from-scratch CNN tower (frame_acc 0.997). Three independent
raw-feature probes (extensions-external, see M1.md) found frozen MuSViT patch
features nearly self-similar across a strip with ~zero distance correlation --
concerning, but not necessarily fatal (D1's own raw CNN features likely lack
distance-correlated structure too, before its positional-encoding+transformer
context stage). This is the real test: train that SAME kind of adapter on top
of frozen MuSViT and see if IT can recover the correct monotone alignment.

MuSViT's tower produces a DIFFERENT column count/spacing than D1's CNN
(w_downsample=4) convention, so onset positions are remapped into MuSViT's own
column space (proportionally, by physical strip-x fraction) rather than reusing
D1Piece's gt_cols directly.

    python -m scripts.overfit_one_piece_m1_musvit --config configs/d1_align_matrix.yaml \
        [--steps 150] [--limit 12]
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
from extensions.alignment.forward_sum import forward_sum_loss
from extensions.alignment.monotonic_decode import viterbi_path
from extensions.alignment.beta_binomial_prior import beta_binomial_log_prior
from mymodel.d1_align_matrix.losses import dense_ce_loss


def build_onset_columns_musvit(piece, w_col_musvit, t_max=None, min_onsets=15):
    """Same GT-path construction as mymodel/m1_monotonic/data.py's
    build_onset_columns, but onset columns are computed in MuSViT's OWN column
    space (proportional strip-x fraction), not D1's w_downsample=4 CNN columns."""
    gt_cols_cnn = piece.gt_cols.numpy().astype(np.int64)
    T = len(gt_cols_cnn)
    if t_max is not None:
        T = min(T, t_max)
    W_scaled = piece.strip.shape[-1]
    wd = piece.w_downsample

    onset_frames = np.asarray(piece.onset_frames, dtype=np.int64)
    onset_frames = np.unique(onset_frames[(onset_frames >= 0) & (onset_frames < T)])
    N = len(onset_frames)
    if N < min_onsets or T < N:
        return None

    x_scaled_px = gt_cols_cnn[onset_frames].astype(np.float64) * wd + wd / 2.0
    onset_cols = np.clip(np.round(x_scaled_px / W_scaled * w_col_musvit).astype(np.int64),
                         0, w_col_musvit - 1)
    # de-duplicate consecutive identical columns is NOT needed -- forward-sum's
    # stay-or-advance-by-one allows repeats naturally as long as monotone.
    onset_cols = np.maximum.accumulate(onset_cols)   # enforce monotone (proportional map is monotone
                                                       # up to rounding ties; this guards exact equality)
    if onset_cols[0] != 0:
        onset_cols = onset_cols - onset_cols[0]
    onset_cols[-1] = max(onset_cols[-1], onset_cols[-2] if N > 1 else 0)

    gt_path = np.searchsorted(onset_frames, np.arange(T), side='right') - 1
    gt_path = np.clip(gt_path, 0, N - 1).astype(np.int64)
    gt_path_cols = onset_cols[gt_path]   # not used directly by forward-sum (needs 0..N-1 indices)
    return T, onset_frames, onset_cols, gt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d1_align_matrix.yaml')
    ap.add_argument('--t_max', type=int, default=300)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--limit', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3.0e-4)
    ap.add_argument('--prior_scale', type=float, default=1.0)
    ap.add_argument('--anneal_frac', type=float, default=0.5)
    ap.add_argument('--ce_weight', type=float, default=1.0)
    ap.add_argument('--ce_sigma_cols', type=float, default=2.0)   # MuSViT cols are coarser -> wider sigma
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = yaml.safe_load(open(a.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dc = cfg['data']

    pieces = d1data.load_split('train', dc['processed_root'], dc['cpjku_data'],
                               dc['mert_roots'], dc['scale_factor'],
                               cfg['model']['w_downsample'], limit=a.limit)
    piece = next((p for p in pieces if p is not None), None)
    assert piece is not None, 'no usable piece found'
    print(f'Testing MuSViT score tower on piece "{piece.piece_name}"  device={device}', flush=True)

    model = M1Model(**{**cfg['model'], 'score_tower': 'musvit'}).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'params: {n_total:,} total, {n_train:,} trainable (MuSViT frozen)', flush=True)

    # Probe MuSViT's own column count once (no grad) to build the onset mapping.
    with torch.no_grad():
        B_probe = model.score_tower(piece.strip.to(device))
    w_col_musvit = B_probe.shape[0]
    print(f'MuSViT score tower produces W_col={w_col_musvit} columns for this piece', flush=True)

    built = build_onset_columns_musvit(piece, w_col_musvit, t_max=a.t_max)
    assert built is not None, 'piece has too few onsets after truncation'
    T, onset_frames, onset_cols, gt_path = built
    N = len(onset_frames)
    print(f'T={T} frames  N={N} onset columns (musvit space, span {onset_cols.min()}..{onset_cols.max()} '
          f'of {w_col_musvit})', flush=True)

    d = np.diff(gt_path)
    assert gt_path[0] == 0 and gt_path[-1] == N - 1, 'GT path endpoints wrong'
    assert set(np.unique(d).tolist()) <= {0, 1}, f'GT path not stay/advance-by-one: {np.unique(d)}'
    print('GT path validated: monotone, step-set', sorted(set(np.unique(d).tolist())), f'0..{N-1}', flush=True)

    mert = piece.mert[:T].to(device)
    strip = piece.strip.to(device)
    cols = torch.from_numpy(onset_cols).to(device)
    gt_path_t = torch.from_numpy(gt_path).to(device)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=a.lr, weight_decay=cfg['optim']['weight_decay'])
    log_prior = beta_binomial_log_prior(T, N, scale=a.prior_scale, device=device)
    anneal_steps = max(1, int(a.anneal_frac * a.steps))

    def decode_accuracy():
        model.eval()
        with torch.no_grad():
            S = model(mert, strip, cols)
            path, _ = viterbi_path(S)
        model.train()
        frame_acc = (path == gt_path_t).float().mean().item()
        col_mae = (path.float() - gt_path_t.float()).abs().mean().item()
        return frame_acc, col_mae

    fa0, mae0 = decode_accuracy()
    print(f'[step   0] (untrained) frame_acc={fa0:.3f}  col_MAE={mae0:.2f}', flush=True)

    t_start = time.time()
    loss0 = None
    for step in range(1, a.steps + 1):
        w = max(0.0, 1.0 - step / anneal_steps)
        S = model(mert, strip, cols)
        S_prior = S + w * log_prior if w > 0 else S
        loss = forward_sum_loss(S_prior, apply_log_softmax=True, normalize=True)
        if a.ce_weight > 0:
            loss = loss + a.ce_weight * dense_ce_loss(S, gt_path_t, sigma_cols=a.ce_sigma_cols)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        opt.step()
        if loss0 is None:
            loss0 = float(loss.detach())
        if step % 15 == 0 or step == a.steps:
            fa, mae = decode_accuracy()
            print(f'[step {step:3d}] loss={float(loss.detach()):.4f} (prior_w={w:.2f})  '
                  f'frame_acc={fa:.3f}  col_MAE={mae:.2f}  {time.time()-t_start:.0f}s', flush=True)

    fa, mae = decode_accuracy()
    print(f'\n=== MuSViT score-tower overfit result ===', flush=True)
    print(f'  loss: {loss0:.4f} -> {float(loss.detach()):.4f}', flush=True)
    print(f'  frame accuracy: {fa0:.3f} -> {fa:.3f}   col_MAE: {mae0:.2f} -> {mae:.2f}', flush=True)
    if fa > 0.90:
        print('  MUSVIT TOWER PASSED overfit bar (matches CNN-tower Phase 1 standard).', flush=True)
    else:
        print('  MUSVIT TOWER DID NOT reach the CNN-tower overfit bar (>0.90) with this budget.', flush=True)


if __name__ == '__main__':
    main()
