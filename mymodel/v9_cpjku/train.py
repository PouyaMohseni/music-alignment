"""v9 training — CPJKU ConditionalUNet with their dice loss and Adam.

Uses random window sampling (simplified from their BPTT approach) since our
strip widths vary between pieces (prevents easy batching for sequential BPTT).
The LSTM receives context within each seq window; works well empirically.

    python -m mymodel.v9_cpjku.train --config configs/v9_cpjku.yaml
"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .cpjku_network import ConditionalUNet
from .data import CPJKUDataset

EPS = 1e-8


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def dice_loss(inputs, targets, smoothing=1.0):
    """Their exact dice loss formula (squared sums, not linear)."""
    iflat = inputs.view(-1)
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smoothing) /
                (iflat.pow(2).sum() + tflat.pow(2).sum() + smoothing + EPS))


def _step(network, b, device):
    # (B, 1, H, W) → (1, B, 1, H, W) for their seq_len=1 forward
    score = b['score_crop'].to(device).unsqueeze(0)  # (1, B, 1, H, W)
    perf  = b['perf'].to(device).unsqueeze(0)        # (1, B, 1, n_mels, n_frames)
    gt    = b['gt_mask'].to(device).unsqueeze(0)     # (1, B, 1, H, W)

    B = score.shape[1]
    hidden = None
    if network.use_lstm:
        hidden = (torch.zeros(network.rnn_layers, B, network.rnn_size).to(device),
                  torch.zeros(network.rnn_layers, B, network.rnn_size).to(device))

    out = network(score=score, perf=perf, hidden=hidden)
    pred = out['segmentation']   # (B, 1, H, W)

    loss = dice_loss(pred, gt.view(B, 1, *gt.shape[-2:]))

    # Accuracy: is predicted peak within 10% of crop width from actual GT position?
    W   = pred.shape[-1]
    col = pred.squeeze(1).sum(dim=1)              # (B, W) — sum over H
    pred_x  = col.argmax(dim=-1)                  # (B,)
    gt_x_local = b['local_gt_x'].to(device)       # (B,)
    acc = ((pred_x - gt_x_local).abs() <= W // 10).float().mean()
    return loss, float(loss.detach()), float(acc.detach())


def _save(network, step, cfg, out_dir):
    path = Path(out_dir) / f"checkpoint_{step:06d}.pt"
    net_config = OmegaConf.to_container(cfg.net)
    torch.save({'step': step, 'state_dict': network.state_dict(),
                'net_config': net_config, 'cfg': OmegaConf.to_container(cfg)}, path)
    return path


@torch.no_grad()
def _validate(network, loader, device):
    network.eval(); losses = []
    for b in loader:
        _, lv, _ = _step(network, b, device)
        losses.append(lv)
    network.train()
    return float(np.mean(losses)) if losses else float('nan')


def main(cfg: DictConfig):
    _seed(cfg.seed)
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(os.getcwd()) / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}  out={out_dir}', flush=True)

    def _loader(split, shuffle):
        ds = CPJKUDataset(
            cfg.data.processed_root, split,
            tile_width=cfg.data.tile_width,
            h_strip=cfg.data.h_strip,
            n_mels=cfg.data.n_mels,
            fps=cfg.data.fps,
            n_frames=cfg.data.n_frames,
            gt_width=cfg.data.gt_width)
        return DataLoader(ds, batch_size=cfg.data.batch_size,
                          shuffle=shuffle, num_workers=cfg.data.num_workers,
                          pin_memory=device == 'cuda')

    tl = _loader('train', shuffle=True)
    try:    vl = _loader('val', shuffle=False)
    except: vl = None
    print(f'train={len(tl.dataset)}  val={len(vl.dataset) if vl else 0}', flush=True)

    # Compute spectrogram stats and set normaliser
    print('Computing spectrogram stats...', flush=True)
    train_ds = tl.dataset
    means, stds = train_ds.compute_spec_stats()
    print(f'  spec means: {means.mean():.3f}  stds: {stds.mean():.3f}', flush=True)

    net_config = OmegaConf.to_container(cfg.net)
    network = ConditionalUNet(net_config)
    network.perf_encoder.set_stats(means, stds)
    network = network.to(device)
    print(f'params: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}',
          flush=True)

    optim = torch.optim.Adam(network.parameters(),
                             lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', patience=cfg.optim.patience, factor=0.5)

    it = iter(tl); t0 = time.time()

    for step in range(1, cfg.train.steps + 1):
        network.train()
        try: b = next(it)
        except StopIteration: it = iter(tl); b = next(it)

        optim.zero_grad(set_to_none=True)
        loss, lv, acc = _step(network, b, device)
        loss.backward()
        if cfg.optim.get('clip_grads'):
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.optim.clip_grads)
        optim.step()

        if step % cfg.train.log_every == 0 or step == 1:
            print(f'step {step:5d}/{cfg.train.steps}  dice={lv:.4f}  '
                  f'acc={acc:.3f}  lr={optim.param_groups[0]["lr"]:.2e}  '
                  f'{time.time()-t0:.1f}s', flush=True)

        if vl and step % cfg.train.eval_every == 0:
            val_loss = _validate(network, vl, device)
            print(f'  [val@{step}] dice={val_loss:.5f}', flush=True)
            scheduler.step(val_loss)

        if step % cfg.train.ckpt_every == 0:
            print(f'  -> saved {_save(network, step, cfg, out_dir)}', flush=True)

    print(f'done in {time.time()-t0:.1f}s', flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/v9_cpjku.yaml')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg


if __name__ == '__main__':
    main(_parse())
