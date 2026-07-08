"""D1 training loop. Whole-piece-per-step with gradient accumulation over
`accum_pieces` pieces per optimizer step (one step already sees hundreds of
frame-level CE terms per piece, so this is NOT CADP's 1-pair-per-step gradient
noise). ReduceLROnPlateau on val loss, early stop at patience*2, checkpoint +
weights-only warm-start/resume.

    python -m mymodel.d1_align_matrix.train --config configs/d1_align_matrix.yaml [--resume]
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

from mymodel.d1_align_matrix.model import D1Model
from mymodel.d1_align_matrix.losses import dense_ce_loss, soft_dtw_matrix_loss
from mymodel.d1_align_matrix import data as d1data


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def piece_loss(model, piece, cfg, device):
    mert = piece.mert.to(device)
    strip = piece.strip.to(device)
    gt_cols = piece.gt_cols.to(device)
    S = model(mert, strip)                         # (T, W_col)
    # guard: model W_col must match gt_cols column range (both from same W, w_downsample)
    W_col = S.shape[1]
    gt_cols = gt_cols.clamp(max=W_col - 1)
    ce = dense_ce_loss(S, gt_cols, sigma_cols=cfg['loss']['ce_sigma_cols'])
    loss = ce
    logs = {'ce': float(ce.detach())}
    if cfg['loss']['dtw_weight'] > 0:
        dtw = soft_dtw_matrix_loss(S, gamma=cfg['loss']['dtw_gamma'],
                                   max_t=cfg['loss']['dtw_max_t'],
                                   max_w=cfg['loss']['dtw_max_w'])
        loss = loss + cfg['loss']['dtw_weight'] * dtw
        logs['dtw'] = float(dtw.detach())
    return loss, logs


def run_epoch(model, pieces, optimizer, cfg, device, train=True, order=None):
    model.train(train)
    accum = cfg['optim']['accum_pieces']
    idxs = list(range(len(pieces)))
    if train and order is not None:
        idxs = order
    total_ce = total_dtw = 0.0
    n = 0
    if train:
        optimizer.zero_grad()
    for step, i in enumerate(idxs):
        piece = pieces[i]
        with torch.set_grad_enabled(train):
            loss, logs = piece_loss(model, piece, cfg, device)
        if train:
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(idxs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
        total_ce += logs['ce']; total_dtw += logs.get('dtw', 0.0); n += 1
    return total_ce / max(n, 1), total_dtw / max(n, 1)


def save_ckpt(path, model, optimizer, scheduler, epoch, best_val, wait):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch,
                'best_val': best_val, 'wait': wait}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d1_align_matrix.yaml')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='limit pieces per split (debug)')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(cfg['train']['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    dc = cfg['data']

    train_pieces = d1data.load_split('train', dc['processed_root'], dc['cpjku_data'],
                                     dc['mert_roots'], dc['scale_factor'],
                                     cfg['model']['w_downsample'], limit=a.limit)
    val_pieces = d1data.load_split('val', dc['processed_root'], dc['cpjku_data'],
                                   dc['mert_roots'], dc['scale_factor'],
                                   cfg['model']['w_downsample'], limit=a.limit)
    if not train_pieces:
        print('ERROR: no training pieces loaded (MERT embeddings missing?)', flush=True)
        sys.exit(1)

    model = D1Model(**cfg['model']).to(device)
    print(f'D1 params: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['optim']['lr'],
                                 weight_decay=cfg['optim']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg['optim']['patience'])

    start_epoch, best_val, wait = 1, float('inf'), 0
    ckpt_latest = out_dir / 'checkpoint_latest.pt'
    if a.resume and ckpt_latest.exists():
        ck = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1; best_val = ck['best_val']; wait = ck['wait']
        print(f'Resumed at epoch {start_epoch} (best_val={best_val:.4f}, wait={wait})', flush=True)

    early_stop = cfg['optim']['patience'] * 2
    rng = random.Random(cfg['seed'])
    for epoch in range(start_epoch, cfg['train']['max_epochs'] + 1):
        t0 = time.time()
        order = list(range(len(train_pieces))); rng.shuffle(order)
        tr_ce, tr_dtw = run_epoch(model, train_pieces, optimizer, cfg, device, train=True, order=order)
        va_ce, va_dtw = run_epoch(model, val_pieces, optimizer, cfg, device, train=False)
        val_loss = va_ce + cfg['loss']['dtw_weight'] * va_dtw
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch {epoch:3d}/{cfg["train"]["max_epochs"]}  '
              f'train_ce={tr_ce:.4f} train_dtw={tr_dtw:.4f}  '
              f'val_ce={va_ce:.4f} val_dtw={va_dtw:.4f}  val_loss={val_loss:.4f}  '
              f'lr={lr:.2e}  {time.time()-t0:.0f}s', flush=True)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss; wait = 0
            save_ckpt(out_dir / 'best_model.pt', model, optimizer, scheduler, epoch, best_val, wait)
            print(f'  -> new best (val={best_val:.5f})', flush=True)
        else:
            wait += 1
            print(f'  no improvement ({wait}/{early_stop})', flush=True)
        save_ckpt(ckpt_latest, model, optimizer, scheduler, epoch, best_val, wait)
        if wait >= early_stop:
            print(f'Early stopping at epoch {epoch}.', flush=True); break

    print('Training done.', flush=True)


if __name__ == '__main__':
    main()
