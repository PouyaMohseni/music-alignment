"""E4 -- multi-tempo D2 training. Reuses train.py's piece_loss/run_epoch/
save_ckpt/set_seed UNCHANGED (they operate on generic D2Piece objects,
tempo-agnostic) -- the only difference is the data-loading call, which now
samples across (piece, tempo_factor) pairs via data_multitempo.load_split_multitempo.

    python -m mymodel.d2_midi_privileged.train_multitempo \
        --config configs/d2_midi_privileged_multitempo.yaml [--resume]
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
from mymodel.d2_midi_privileged.midi_encoder import MidiEncoder
from mymodel.d2_midi_privileged.train import set_seed, run_epoch, save_ckpt
from mymodel.d2_midi_privileged import data_multitempo as d2mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d2_midi_privileged_multitempo.yaml')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np_rng = np.random.default_rng(cfg['seed'])

    out_dir = Path(cfg['train']['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    dc = cfg['data']
    tempo_factors = tuple(dc['tempo_factors'])

    train_pieces = d2mt.load_split_multitempo(
        'train', dc['processed_root'], dc['cpjku_data'], dc['mert_roots'],
        dc['mert_roots_multitempo'], dc['multitempo_render_dir'], dc['scale_factor'],
        cfg['model']['w_downsample'], tempo_factors=tempo_factors, repeat_k=dc['repeat_k'],
        limit=a.limit)
    # val stays single-tempo (tempo_1000) -- model selection should reflect the
    # real eval distribution, not be inflated by matching train-time tempo variety.
    val_pieces = d2mt.load_split_multitempo(
        'val', dc['processed_root'], dc['cpjku_data'], dc['mert_roots'],
        dc['mert_roots_multitempo'], dc['multitempo_render_dir'], dc['scale_factor'],
        cfg['model']['w_downsample'], tempo_factors=(1000,), repeat_k=dc['repeat_k'],
        limit=a.limit)
    if not train_pieces:
        print('ERROR: no training pieces loaded', flush=True); sys.exit(1)

    model = D1Model(**cfg['model']).to(device)
    midi_encoder = MidiEncoder(d_model=cfg['model']['d_model']).to(device)
    print(f'D1Model params: {sum(p.numel() for p in model.parameters()):,}  '
          f'MidiEncoder params: {sum(p.numel() for p in midi_encoder.parameters()):,} '
          f'(train-only, discarded at inference)  tempo_factors={tempo_factors}', flush=True)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(midi_encoder.parameters()),
        lr=cfg['optim']['lr'], weight_decay=cfg['optim']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg['optim']['patience'])

    start_epoch, best_val, wait = 1, float('inf'), 0
    ckpt_latest = out_dir / 'checkpoint_latest.pt'
    if a.resume and ckpt_latest.exists():
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
