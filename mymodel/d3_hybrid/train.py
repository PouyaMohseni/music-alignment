"""D3 training loop -- identical to D2's (mymodel/d2_midi_privileged/train.py)
except D3Model (hybrid audio tower) replaces D1Model, and the audio tower is
warm-started from v13's trained MERTProjector weights before training starts
(see model.py's HybridAudioTower.load_pretrained_v13 for why warm-start, not
scratch: isolates "does D2's decode/training-signal advantage compound with a
BETTER audio representation" as the only variable under test).

    python -m mymodel.d3_hybrid.train --config configs/d3_hybrid.yaml [--resume]
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

from mymodel.d3_hybrid.model import D3Model
from mymodel.d1_align_matrix.losses import soft_dtw_matrix_loss
from mymodel.d2_midi_privileged.midi_encoder import MidiEncoder
from mymodel.d2_midi_privileged.losses import soft_multi_target_ce_loss, midi_distill_loss
from mymodel.d2_midi_privileged import data as d2data

MAX_DISTILL_FRAMES = 512


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def piece_loss(model, midi_encoder, piece, cfg, device, rng):
    mert = piece.mert.to(device)
    strip = piece.strip.to(device)
    gt_cols = piece.gt_cols.to(device)
    A, B = model.encode(mert, strip)
    S = model.similarity(A, B)
    W_col = S.shape[1]
    gt_cols = gt_cols.clamp(max=W_col - 1)

    ce = soft_multi_target_ce_loss(S, gt_cols, repeat_alt_cols=piece.repeat_alt_cols,
                                   sigma_cols=cfg['loss']['ce_sigma_cols'],
                                   alt_weight=cfg['loss']['repeat_alt_weight'])
    loss = ce
    logs = {'ce': float(ce.detach())}

    if cfg['loss']['dtw_weight'] > 0:
        dtw = soft_dtw_matrix_loss(S, gamma=cfg['loss']['dtw_gamma'],
                                   max_t=cfg['loss']['dtw_max_t'], max_w=cfg['loss']['dtw_max_w'])
        loss = loss + cfg['loss']['dtw_weight'] * dtw
        logs['dtw'] = float(dtw.detach())

    if cfg['loss']['distill_weight'] > 0:
        T = A.shape[0]
        pitch_roll = piece.pitch_roll.to(device)
        if T > MAX_DISTILL_FRAMES:
            idx = torch.from_numpy(rng.choice(T, size=MAX_DISTILL_FRAMES, replace=False)).to(device)
            A_sub, pr_sub = A[idx], pitch_roll[idx]
        else:
            A_sub, pr_sub = A, pitch_roll
        M = midi_encoder(pr_sub)
        distill = midi_distill_loss(A_sub, M, temperature=cfg['loss']['distill_temperature'])
        loss = loss + cfg['loss']['distill_weight'] * distill
        logs['distill'] = float(distill.detach())

    return loss, logs


def run_epoch(model, midi_encoder, pieces, optimizer, cfg, device, train, order, np_rng):
    model.train(train); midi_encoder.train(train)
    accum = cfg['optim']['accum_pieces']
    idxs = order if (train and order is not None) else list(range(len(pieces)))
    totals = {'ce': 0.0, 'dtw': 0.0, 'distill': 0.0}
    n = 0
    if train:
        optimizer.zero_grad()
    for step, i in enumerate(idxs):
        piece = pieces[i]
        with torch.set_grad_enabled(train):
            loss, logs = piece_loss(model, midi_encoder, piece, cfg, device, np_rng)
        if train:
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(idxs):
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(midi_encoder.parameters()), 5.0)
                optimizer.step()
                optimizer.zero_grad()
        for k in totals:
            totals[k] += logs.get(k, 0.0)
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def save_ckpt(path, model, midi_encoder, optimizer, scheduler, epoch, best_val, wait):
    torch.save({'model': model.state_dict(), 'midi_encoder': midi_encoder.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'best_val': best_val, 'wait': wait}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d3_hybrid.yaml')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np_rng = np.random.default_rng(cfg['seed'])

    out_dir = Path(cfg['train']['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    dc = cfg['data']

    train_pieces = d2data.load_split('train', dc['processed_root'], dc['cpjku_data'],
                                     dc['mert_roots'], dc['scale_factor'],
                                     cfg['model']['w_downsample'], repeat_k=dc['repeat_k'],
                                     limit=a.limit)
    val_pieces = d2data.load_split('val', dc['processed_root'], dc['cpjku_data'],
                                   dc['mert_roots'], dc['scale_factor'],
                                   cfg['model']['w_downsample'], repeat_k=dc['repeat_k'],
                                   limit=a.limit)
    if not train_pieces:
        print('ERROR: no training pieces loaded', flush=True); sys.exit(1)

    model_kwargs = {k: v for k, v in cfg['model'].items() if k != 'warm_start_ckpt'}
    model = D3Model(**model_kwargs).to(device)
    midi_encoder = MidiEncoder(d_model=cfg['model']['d_model']).to(device)

    ckpt_latest = out_dir / 'checkpoint_latest.pt'
    resumed = a.resume and ckpt_latest.exists()
    warm_start_ckpt = cfg['model'].get('warm_start_ckpt')
    if warm_start_ckpt and not resumed:
        missing, unexpected = model.audio_tower.load_pretrained_v13(warm_start_ckpt, device=device)
        print(f'Warm-started audio tower from {warm_start_ckpt} '
              f'(missing={missing}, unexpected={unexpected})', flush=True)

    print(f'D3Model params: {sum(p.numel() for p in model.parameters()):,}  '
          f'MidiEncoder params: {sum(p.numel() for p in midi_encoder.parameters()):,} '
          f'(train-only, discarded at inference)', flush=True)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(midi_encoder.parameters()),
        lr=cfg['optim']['lr'], weight_decay=cfg['optim']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg['optim']['patience'])

    start_epoch, best_val, wait = 1, float('inf'), 0
    if resumed:
        ck = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ck['model']); midi_encoder.load_state_dict(ck['midi_encoder'])
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1; best_val = ck['best_val']; wait = ck['wait']
        print(f'Resumed at epoch {start_epoch} (best_val={best_val:.4f}, wait={wait})', flush=True)

    early_stop = cfg['optim']['patience'] * 2
    rng = random.Random(cfg['seed'])
    for epoch in range(start_epoch, cfg['train']['max_epochs'] + 1):
        t0 = time.time()
        order = list(range(len(train_pieces))); rng.shuffle(order)
        tr = run_epoch(model, midi_encoder, train_pieces, optimizer, cfg, device, True, order, np_rng)
        va = run_epoch(model, midi_encoder, val_pieces, optimizer, cfg, device, False, None, np_rng)
        val_loss = (va['ce'] + cfg['loss']['dtw_weight'] * va['dtw']
                   + cfg['loss']['distill_weight'] * va['distill'])
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch {epoch:3d}/{cfg["train"]["max_epochs"]}  '
              f'train_ce={tr["ce"]:.4f} train_dtw={tr["dtw"]:.4f} train_distill={tr["distill"]:.4f}  '
              f'val_ce={va["ce"]:.4f} val_dtw={va["dtw"]:.4f} val_distill={va["distill"]:.4f}  '
              f'val_loss={val_loss:.4f}  lr={lr:.2e}  {time.time()-t0:.0f}s', flush=True)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss; wait = 0
            save_ckpt(out_dir / 'best_model.pt', model, midi_encoder, optimizer, scheduler,
                      epoch, best_val, wait)
            print(f'  -> new best (val={best_val:.5f})', flush=True)
        else:
            wait += 1
            print(f'  no improvement ({wait}/{early_stop})', flush=True)
        save_ckpt(ckpt_latest, model, midi_encoder, optimizer, scheduler, epoch, best_val, wait)
        if wait >= early_stop:
            print(f'Early stopping at epoch {epoch}.', flush=True); break

    print('Training done.', flush=True)


if __name__ == '__main__':
    main()
