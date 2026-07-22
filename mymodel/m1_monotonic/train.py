"""M1 Phase 2 -- full training. Mirrors D1's whole-piece training loop
(mymodel/d1_align_matrix/train.py: whole-piece-per-step, gradient accumulation,
ReduceLROnPlateau, best-by-val + early stop, weights-only resume) with ONE
objective swap: forward-sum monotonic-alignment loss over onset columns
(extensions/alignment/forward_sum.py) + an annealed beta-binomial prior, instead
of D1's dense per-frame CE + soft-DTW.

Best checkpoint is selected by val forward-sum loss -- unlike dice loss (which
this project documented does NOT track pct@0.5s 1:1), the forward-sum loss IS
the monotonic-alignment likelihood the Viterbi decoder maximises, so lower val
forward-sum is a direct proxy for decode quality. Val Viterbi frame-accuracy is
also logged for monitoring.

    python -m mymodel.m1_monotonic.train --config configs/m1_monotonic.yaml [--resume]
"""
from __future__ import annotations
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mymodel.d1_align_matrix import data as d1data
from mymodel.m1_monotonic.model import M1Model
from mymodel.m1_monotonic.data import build_onset_columns
from extensions.alignment.forward_sum import forward_sum_loss
from extensions.alignment.monotonic_decode import viterbi_path
from extensions.alignment.beta_binomial_prior import beta_binomial_prior
from mymodel.d1_align_matrix.losses import dense_ce_loss


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def precompute(pieces, w_downsample, t_max, with_prior, prior_scale):
    """Attach the onset-column representation (and optional beta-binomial log
    prior) to each usable piece. Returns list of dicts; drops pieces with too
    few onsets. Prior is cached on CPU (moved to device per step only while its
    anneal weight > 0)."""
    out = []
    for p in pieces:
        built = build_onset_columns(p, t_max=t_max)
        if built is None:
            continue
        T, onset_frames, onset_cols, onset_x, gt_path = built
        N = len(onset_frames)
        log_prior = None
        if with_prior:
            P = beta_binomial_prior(T, N, scale=prior_scale)          # (T, N) prob
            log_prior = torch.from_numpy(np.log(np.clip(P, 1e-12, None)).astype(np.float32))
        out.append({
            'name': p.piece_name, 'mert': p.mert[:T], 'strip': p.strip, 'T': T,
            'onset_cols': torch.from_numpy(onset_cols),
            'gt_path': torch.from_numpy(gt_path),
            'onset_frames': onset_frames, 'log_prior': log_prior,
        })
    return out


def piece_loss(model, pc, device, prior_weight, ce_weight, ce_sigma, fs_weight):
    """M1 objective: CE content/timing anchor + forward-sum monotonic-path
    regularizer + annealed beta-binomial prior. Forward-sum alone is
    underdetermined (Phase-1 overfit: loss->0 on a valid-but-wrong monotone
    path); CE anchors each frame to its GT onset column."""
    S = model(pc['mert'].to(device), pc['strip'].to(device), pc['onset_cols'].to(device))
    S_prior = S + prior_weight * pc['log_prior'].to(device) \
        if (prior_weight > 0 and pc['log_prior'] is not None) else S
    loss = fs_weight * forward_sum_loss(S_prior, apply_log_softmax=True, normalize=True)
    if ce_weight > 0:
        loss = loss + ce_weight * dense_ce_loss(S, pc['gt_path'].to(device), sigma_cols=ce_sigma)
    return loss


@torch.no_grad()
def val_frame_acc(model, pcs, device):
    model.eval()
    accs = []
    for pc in pcs:
        S = model(pc['mert'].to(device), pc['strip'].to(device), pc['onset_cols'].to(device))
        path, _ = viterbi_path(S)
        accs.append((path == pc['gt_path'].to(device)).float().mean().item())
    model.train()
    return float(np.mean(accs)) if accs else 0.0


def run_epoch(model, pcs, optimizer, accum, device, prior_weight,
              ce_weight, ce_sigma, fs_weight, train=True, order=None):
    model.train(train)
    idxs = order if (train and order is not None) else list(range(len(pcs)))
    total = 0.0; n = 0
    if train:
        optimizer.zero_grad()
    for step, i in enumerate(idxs):
        with torch.set_grad_enabled(train):
            loss = piece_loss(model, pcs[i], device, prior_weight, ce_weight, ce_sigma, fs_weight)
        if train:
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(idxs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); optimizer.zero_grad()
        total += float(loss.detach()); n += 1
    return total / max(n, 1)


def save_ckpt(path, model, optimizer, scheduler, epoch, best_val, wait):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch,
                'best_val': best_val, 'wait': wait}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/m1_monotonic.yaml')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cfg['train']['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    dc = cfg['data']; lc = cfg['loss']
    wd = cfg['model']['w_downsample']

    prior_epochs = lc.get('prior_anneal_epochs', 0)
    with_prior = prior_epochs > 0
    t_max = cfg['train'].get('max_train_frames', None)

    tr = d1data.load_split('train', dc['processed_root'], dc['cpjku_data'], dc['mert_roots'],
                           dc['scale_factor'], wd, limit=a.limit)
    va = d1data.load_split('val', dc['processed_root'], dc['cpjku_data'], dc['mert_roots'],
                           dc['scale_factor'], wd, limit=a.limit)
    train_pcs = precompute(tr, wd, t_max, with_prior, lc.get('prior_scale', 1.0))
    val_pcs = precompute(va, wd, t_max, False, 1.0)
    if not train_pcs:
        print('ERROR: no training pieces', flush=True); sys.exit(1)
    print(f'[M1] train={len(train_pcs)} val={len(val_pcs)} pieces  '
          f'prior_anneal_epochs={prior_epochs}  max_train_frames={t_max}', flush=True)

    model = M1Model(**cfg['model']).to(device)
    print(f'[M1] params: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['optim']['lr'],
                                 weight_decay=cfg['optim']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg['optim']['patience'])

    # Best checkpoint + early stop track val Viterbi FRAME ACCURACY (maximize) --
    # the direct decode-quality proxy -- not the loss, per this project's
    # documented dice-loss-vs-pct@0.5s mismatch lesson. Scheduler steps on the
    # smooth val loss (minimize).
    ce_w, ce_sig, fs_w = lc.get('ce_weight', 1.0), lc.get('ce_sigma_cols', 1.0), lc.get('fs_weight', 1.0)
    start_epoch, best_acc, wait = 1, -1.0, 0
    ckpt_latest = out_dir / 'checkpoint_latest.pt'
    if a.resume and ckpt_latest.exists():
        ck = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1; best_acc = ck['best_val']; wait = ck['wait']
        print(f'[M1] resumed at epoch {start_epoch} (best_val_acc={best_acc:.4f}, wait={wait})', flush=True)

    early_stop = cfg['optim']['patience'] * 2
    rng = random.Random(cfg['seed'])
    for epoch in range(start_epoch, cfg['train']['max_epochs'] + 1):
        t0 = time.time()
        prior_weight = max(0.0, 1.0 - (epoch - 1) / prior_epochs) if with_prior else 0.0
        order = list(range(len(train_pcs))); rng.shuffle(order)
        tr_loss = run_epoch(model, train_pcs, optimizer, cfg['optim']['accum_pieces'],
                            device, prior_weight, ce_w, ce_sig, fs_w, train=True, order=order)
        va_loss = run_epoch(model, val_pcs, optimizer, cfg['optim']['accum_pieces'],
                            device, 0.0, ce_w, ce_sig, fs_w, train=False)
        va_acc = val_frame_acc(model, val_pcs, device)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch {epoch:3d}/{cfg["train"]["max_epochs"]}  train_loss={tr_loss:.4f}  '
              f'val_loss={va_loss:.4f}  val_frame_acc={va_acc:.3f}  prior_w={prior_weight:.2f}  '
              f'lr={lr:.2e}  {time.time()-t0:.0f}s', flush=True)

        is_best = va_acc > best_acc
        if is_best:
            best_acc = va_acc; wait = 0
            save_ckpt(out_dir / 'best_model.pt', model, optimizer, scheduler, epoch, best_acc, wait)
            print(f'  -> new best (val_frame_acc={best_acc:.4f})', flush=True)
        else:
            wait += 1
            print(f'  no improvement ({wait}/{early_stop})', flush=True)
        save_ckpt(ckpt_latest, model, optimizer, scheduler, epoch, best_acc, wait)
        if wait >= early_stop:
            print(f'Early stopping at epoch {epoch}.', flush=True); break

    print('M1 training done.', flush=True)


if __name__ == '__main__':
    main()
