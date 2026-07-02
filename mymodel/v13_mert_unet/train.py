"""v13/v14/v15: Full-strip BPTT training with pre-computed MERT audio features.

Three variants distinguished by config audio_encoder field:
  v13  audio_encoder: MERTProjector   n_frames: 1  (single MERT frame → Linear)
  v14  audio_encoder: MERTBiLSTM      n_frames: 8  (8-frame window → BiLSTM)
  v15  audio_encoder: MERTMlpProjector n_frames: 1  (single MERT frame → MLP)

Usage:
    python -m mymodel.v13_mert_unet.train --config configs/v13_mert_linear.yaml
"""
from __future__ import annotations
import argparse, os, random, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from .data import FullStripDataset, make_gt_mask

EPS = 1e-8


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def dice_loss(inputs, targets, smoothing=0.0):
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smoothing) /
                (iflat.pow(2).sum() + tflat.pow(2).sum() + smoothing + EPS))


def _build_perf(feats: np.ndarray, frames: list[int], n_frames: int) -> np.ndarray:
    """Build (sl, 1, 768, n_frames) perf array from MERT features.

    feats: (T, 768) pre-computed MERT features at 20fps.
    For n_frames=1: grabs single frame. For n_frames=8: grabs 8-frame sliding window.
    Layout mirrors CBEncoder's (sl, 1, n_mels, n_frames) convention.
    """
    T, mert_dim = feats.shape
    sl = len(frames)
    out = np.zeros((sl, 1, mert_dim, n_frames), dtype=np.float32)
    for i, t in enumerate(frames):
        t = min(t, T - 1)
        t0 = max(0, t - n_frames + 1)
        window = feats[t0:t + 1]             # (actual_len, 768)
        actual_len = window.shape[0]
        if actual_len < n_frames:            # left-pad at start of piece
            pad = np.zeros((n_frames - actual_len, mert_dim), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        out[i, 0] = window.T                 # (768, n_frames)
    return out


def _build_gt(H, W_sc, strip_x_sc, frames, gt_width):
    sl = len(frames)
    out = np.zeros((sl, 1, H, W_sc), dtype=np.float32)
    for i, t in enumerate(frames):
        cx = int(np.clip(round(strip_x_sc[t]), 0, W_sc - 1))
        out[i, 0] = make_gt_mask(H, W_sc, cx=cx, gt_width=gt_width)
    return out


def _init_hidden(network, device):
    if not network.use_lstm:
        return None
    return (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
            torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))


def _detach_hidden(hidden):
    return tuple(h.detach() for h in hidden) if hidden else None


def _forward_chunk(network, score_1frame, feats, strip_x_sc,
                   frames, n_frames, gt_width, hidden, device):
    sl   = len(frames)
    H    = score_1frame.shape[-2]
    W_sc = score_1frame.shape[-1]

    score_batch = score_1frame.expand(sl, -1, -1, -1, -1)

    perf_np    = _build_perf(feats, frames, n_frames)
    perf_batch = torch.from_numpy(perf_np).to(device).unsqueeze(1)  # (sl,1,1,768,n_frames)

    gt_np    = _build_gt(H, W_sc, strip_x_sc, frames, gt_width)
    gt_batch = torch.from_numpy(gt_np).to(device)

    out      = network(score=score_batch, perf=perf_batch, hidden=hidden)
    pred     = out['segmentation']
    new_hidden = _detach_hidden(out.get('hidden'))

    loss = dice_loss(pred, gt_batch, smoothing=0.0)

    with torch.no_grad():
        col    = pred.squeeze(1).sum(dim=1)
        pred_x = col.argmax(dim=-1).float()
        gt_x   = torch.tensor([strip_x_sc[f] for f in frames],
                               device=device, dtype=torch.float32)
        acc = ((pred_x - gt_x).abs() <= W_sc * 0.1).float().mean()

    return loss, float(acc), new_hidden


def _train_epoch(network, dataset, optimizer, seq_len, n_frames, gt_width, device):
    network.train()
    order = list(range(len(dataset)))
    np.random.shuffle(order)
    total_loss = total_acc = n_seen = 0

    for idx in order:
        p       = dataset[idx]
        score_1 = torch.from_numpy(p['score'][np.newaxis, np.newaxis, np.newaxis]).to(device)
        feats   = p['feats']
        T       = p['T']

        hidden = _init_hidden(network, device)
        t = 0
        while t < T:
            frames = list(range(t, min(t + seq_len, T)))
            sl     = len(frames)
            optimizer.zero_grad(set_to_none=True)
            loss, acc, hidden = _forward_chunk(
                network, score_1, feats, p['strip_x_sc'],
                frames, n_frames, gt_width, hidden, device)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * sl
            total_acc  += acc * sl
            n_seen     += sl
            t          += sl

    denom = max(1, n_seen)
    return total_loss / denom, total_acc / denom


@torch.no_grad()
def _val_epoch(network, dataset, seq_len, n_frames, gt_width, device):
    network.eval()
    total_loss = n_seen = 0
    for idx in range(len(dataset)):
        p       = dataset[idx]
        score_1 = torch.from_numpy(p['score'][np.newaxis, np.newaxis, np.newaxis]).to(device)
        feats   = p['feats']
        T       = p['T']
        hidden  = _init_hidden(network, device)
        t = 0
        while t < T:
            frames = list(range(t, min(t + seq_len, T)))
            loss, _, hidden = _forward_chunk(
                network, score_1, feats, p['strip_x_sc'],
                frames, n_frames, gt_width, hidden, device)
            total_loss += float(loss) * len(frames)
            n_seen     += len(frames)
            t          += len(frames)
    network.train()
    return total_loss / max(1, n_seen)


def _save(network, optimizer, scheduler, epoch, best_val, wait, cfg, out_dir, is_best):
    payload = {
        'epoch': epoch, 'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'best_val_loss': best_val, 'wait': wait,
        'net_config': OmegaConf.to_container(cfg.net),
        'cfg': OmegaConf.to_container(cfg),
    }
    path = out_dir / f'checkpoint_epoch{epoch:03d}.pt'
    torch.save(payload, path)
    if is_best:
        torch.save(payload, out_dir / 'best_model.pt')
    old = out_dir / f'checkpoint_epoch{epoch - 2:03d}.pt'
    if old.exists():
        old.unlink()
    return path


def main(cfg: DictConfig, resume: str | None = None):
    _seed(cfg.seed)
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(os.getcwd()) / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}  out={out_dir}', flush=True)

    train_ds = FullStripDataset(cfg.data.processed_root, cfg.data.mert_emb_root,
                                'train', h_strip=cfg.data.h_strip,
                                w_scale=cfg.data.w_scale, fps=cfg.data.fps)
    val_ds   = FullStripDataset(cfg.data.processed_root, cfg.data.mert_emb_root,
                                'val',   h_strip=cfg.data.h_strip,
                                w_scale=cfg.data.w_scale, fps=cfg.data.fps)
    print(f'train={len(train_ds)}  val={len(val_ds)}', flush=True)

    net_config = OmegaConf.to_container(cfg.net)
    network    = ConditionalUNet(net_config)
    network.perf_encoder.set_stats(None, None)   # no-op for MERT encoders
    network    = network.to(device)
    print(f'params: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}',
          flush=True)

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.optim.patience, factor=0.5)

    max_epochs     = cfg.train.max_epochs
    early_patience = cfg.optim.patience * 2
    best_val       = float('inf')
    wait           = 0
    start_epoch    = 1

    ckpt_to_load = None
    if resume:
        ckpt_to_load = resume if resume != 'auto' else None
        if ckpt_to_load is None:
            candidates = sorted(out_dir.glob('checkpoint_epoch*.pt'))
            if candidates:
                ckpt_to_load = str(candidates[-1])
                print(f'Auto-resume: {ckpt_to_load}', flush=True)

    if ckpt_to_load and Path(ckpt_to_load).exists():
        ckpt = torch.load(ckpt_to_load, map_location='cpu', weights_only=False)
        network.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_val    = ckpt['best_val_loss']
        wait        = ckpt['wait']
        start_epoch = ckpt['epoch'] + 1
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f'Resumed epoch={ckpt["epoch"]}  best={best_val:.5f}  wait={wait}',
              flush=True)

    t0 = time.time()
    n_frames = cfg.data.n_frames
    gt_width = cfg.data.gt_width

    for epoch in range(start_epoch, max_epochs + 1):
        tr_loss, tr_acc = _train_epoch(
            network, train_ds, optimizer,
            seq_len=cfg.train.seq_len, n_frames=n_frames,
            gt_width=gt_width, device=device)
        val_loss = _val_epoch(
            network, val_ds,
            seq_len=cfg.train.seq_len, n_frames=n_frames,
            gt_width=gt_width, device=device)
        scheduler.step(val_loss)

        is_best = val_loss < best_val
        print(f'epoch {epoch:3d}/{max_epochs}  '
              f'train_dice={tr_loss:.4f}  acc={tr_acc:.3f}  '
              f'val_dice={val_loss:.4f}  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
              f'{time.time()-t0:.0f}s', flush=True)

        if is_best:
            best_val = val_loss
            wait     = 0
            _save(network, optimizer, scheduler, epoch,
                  best_val, wait, cfg, out_dir, is_best=True)
            print(f'  -> new best (val={best_val:.5f})', flush=True)
        else:
            wait += 1
            _save(network, optimizer, scheduler, epoch,
                  best_val, wait, cfg, out_dir, is_best=False)
            print(f'  no improvement ({wait}/{early_patience})', flush=True)
            if wait >= early_patience:
                print(f'Early stopping at epoch {epoch}.', flush=True)
                break

    print(f'Done. best_val_loss={best_val:.5f}', flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', nargs='?', const='auto', default=None)
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg, a.resume


if __name__ == '__main__':
    cfg, resume = _parse()
    main(cfg, resume=resume)
