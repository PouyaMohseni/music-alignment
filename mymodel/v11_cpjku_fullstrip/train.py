"""v11: BPTT training matching CPJKU iterate_dataset exactly.

Their approach:
  - Full score page per frame (we use full strip) → GT at TRUE position
  - BPTT with seq_len=16 chunks, LSTM state maintained across chunks
  - batch_size=1 (our strips have variable widths, can't mix pieces)
  - 100 epochs max, ReduceLROnPlateau (patience=5, factor=0.5)
  - Early stopping when patience×2=10 epochs without val_loss improvement
  - dice smoothing=0 (their exact iterate_dataset call)
  - Spectrogram normalisation computed from training set before training

    python -m mymodel.v11_cpjku_fullstrip.train --config configs/v11_cpjku_fullstrip.yaml
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


def _seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor,
              smoothing: float = 0.0) -> torch.Tensor:
    """Their exact dice loss with smoothing=0 (iterate_dataset call)."""
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smoothing) /
                (iflat.pow(2).sum() + tflat.pow(2).sum() + smoothing + EPS))


def _build_perf(spec: np.ndarray, frames: list[int], n_frames: int) -> np.ndarray:
    """Build (sl, 1, n_mels, n_frames) perf array for a list of frame indices.

    Shapes match CBEncoder.reshape_input which expects (seq_len, bs, c, h, w):
    we add the bs=1 dimension at the call site with unsqueeze(1).
    """
    n_mels = spec.shape[0]
    sl = len(frames)
    out = np.zeros((sl, 1, n_mels, n_frames), dtype=np.float32)
    for i, t in enumerate(frames):
        t0 = max(0, t - n_frames)
        window = spec[:, t0:t]
        if window.shape[-1] < n_frames:
            window = np.pad(window, ((0, 0), (n_frames - window.shape[-1], 0)))
        out[i, 0] = window
    return out  # (sl, c=1, n_mels, n_frames)


def _build_gt(H: int, W_sc: int, strip_x_sc: np.ndarray,
              frames: list[int], gt_width: int) -> np.ndarray:
    """Build (sl, 1, H, W_sc) GT mask array for a list of frame indices."""
    sl = len(frames)
    out = np.zeros((sl, 1, H, W_sc), dtype=np.float32)
    for i, t in enumerate(frames):
        cx = int(np.clip(round(strip_x_sc[t]), 0, W_sc - 1))
        out[i, 0] = make_gt_mask(H, W_sc, cx=cx, gt_width=gt_width)
    return out  # (sl, c=1, H, W_sc)


def _init_hidden(network: ConditionalUNet, device: str):
    if not network.use_lstm:
        return None
    return (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
            torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))


def _detach_hidden(hidden):
    if hidden is None:
        return None
    return tuple(h.detach() for h in hidden)


def _forward_chunk(network, score_1frame, spec, strip_x_sc,
                   frames, n_frames, gt_width, hidden, device):
    """One BPTT chunk: forward + loss. Returns (loss, acc, new_hidden).

    score_1frame: (1, 1, 1, H, W_sc) — a single-frame score expanded in forward
    Network expects score: (sl, bs=1, c=1, H, W_sc)
    Network expects perf:  (sl, bs=1, c=1, n_mels, n_frames)
    Network output seg:    (sl*bs=sl, c=1, H, W_sc)
    """
    sl = len(frames)
    H  = score_1frame.shape[-2]
    W_sc = score_1frame.shape[-1]

    # score_batch: (sl, 1, 1, H, W_sc) — expand the stored (1,1,1,H,W) tile
    score_batch = score_1frame.expand(sl, -1, -1, -1, -1)

    # perf_batch: (sl, bs=1, c=1, n_mels, n_frames)
    perf_np = _build_perf(spec, frames, n_frames)        # (sl, c=1, n_mels, n_frames)
    perf_batch = torch.from_numpy(perf_np).to(device)
    perf_batch = perf_batch.unsqueeze(1)                 # (sl, bs=1, c=1, n_mels, n_frames)

    # gt_batch: (sl, 1, H, W_sc) — matches segmentation output shape
    gt_np = _build_gt(H, W_sc, strip_x_sc, frames, gt_width)
    gt_batch = torch.from_numpy(gt_np).to(device)        # (sl, 1, H, W_sc)

    out  = network(score=score_batch, perf=perf_batch, hidden=hidden)
    pred = out['segmentation']   # (sl, 1, H, W_sc)
    new_hidden = _detach_hidden(out.get('hidden'))

    loss = dice_loss(pred, gt_batch, smoothing=0.0)

    # Accuracy: column argmax vs true GT x
    with torch.no_grad():
        col    = pred.squeeze(1).sum(dim=1)              # (sl, W_sc)
        pred_x = col.argmax(dim=-1).float()              # (sl,)
        gt_x   = torch.tensor([strip_x_sc[f] for f in frames],
                              device=device, dtype=torch.float32)
        acc    = ((pred_x - gt_x).abs() <= W_sc * 0.1).float().mean()

    return loss, float(acc), new_hidden


def _train_epoch(network: ConditionalUNet, dataset: FullStripDataset,
                 optimizer: torch.optim.Optimizer,
                 seq_len: int, n_frames: int, gt_width: int,
                 device: str) -> tuple[float, float]:
    """One training epoch: process all pieces in random order with BPTT."""
    network.train()
    order = list(range(len(dataset)))
    np.random.shuffle(order)

    total_loss = total_acc = n_frames_seen = 0

    for idx in order:
        piece      = dataset[idx]
        score      = piece['score']       # (H, W_sc)
        spec       = piece['spec']        # (n_mels, T)
        strip_x_sc = piece['strip_x_sc']  # (T,)
        T          = piece['T']

        # Store score once on device as (seq=1, bs=1, c=1, H, W_sc)
        # expand() in _forward_chunk will broadcast to (sl, 1, 1, H, W_sc)
        score_1 = torch.from_numpy(
            score[np.newaxis, np.newaxis, np.newaxis]   # (1, 1, 1, H, W_sc)
        ).to(device, non_blocking=True)

        hidden = _init_hidden(network, device)
        t = 0

        while t < T:
            frames = list(range(t, min(t + seq_len, T)))
            sl     = len(frames)

            optimizer.zero_grad(set_to_none=True)
            loss, acc, hidden = _forward_chunk(
                network, score_1, spec, strip_x_sc,
                frames, n_frames, gt_width, hidden, device)

            loss.backward()
            optimizer.step()

            total_loss    += float(loss.detach()) * sl
            total_acc     += acc * sl
            n_frames_seen += sl
            t             += sl

    denom = max(1, n_frames_seen)
    return total_loss / denom, total_acc / denom


@torch.no_grad()
def _val_epoch(network: ConditionalUNet, dataset: FullStripDataset,
               seq_len: int, n_frames: int, gt_width: int,
               device: str) -> float:
    """One validation epoch. Returns mean dice loss over all frames."""
    network.eval()
    total_loss = n_frames_seen = 0

    for idx in range(len(dataset)):
        piece      = dataset[idx]
        score      = piece['score']
        spec       = piece['spec']
        strip_x_sc = piece['strip_x_sc']
        T          = piece['T']

        score_1 = torch.from_numpy(
            score[np.newaxis, np.newaxis, np.newaxis]
        ).to(device, non_blocking=True)

        hidden = _init_hidden(network, device)
        t = 0

        while t < T:
            frames = list(range(t, min(t + seq_len, T)))
            sl     = len(frames)

            loss, _, hidden = _forward_chunk(
                network, score_1, spec, strip_x_sc,
                frames, n_frames, gt_width, hidden, device)

            total_loss    += float(loss) * sl
            n_frames_seen += sl
            t             += sl

    network.train()
    return total_loss / max(1, n_frames_seen)


def _save_best(network: ConditionalUNet, epoch: int, cfg: DictConfig,
               out_dir: Path) -> Path:
    ckpt_path = out_dir / f'checkpoint_epoch{epoch:03d}.pt'
    best_path  = out_dir / 'best_model.pt'
    payload = {
        'epoch':      epoch,
        'state_dict': network.state_dict(),
        'net_config': OmegaConf.to_container(cfg.net),
        'cfg':        OmegaConf.to_container(cfg),
    }
    torch.save(payload, ckpt_path)
    torch.save(payload, best_path)
    return ckpt_path


def main(cfg: DictConfig):
    _seed(cfg.seed)
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(os.getcwd()) / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}  out={out_dir}', flush=True)

    train_ds = FullStripDataset(
        cfg.data.processed_root, 'train',
        h_strip=cfg.data.h_strip, w_scale=cfg.data.w_scale,
        n_mels=cfg.data.n_mels, fps=cfg.data.fps)

    val_ds = FullStripDataset(
        cfg.data.processed_root, 'val',
        h_strip=cfg.data.h_strip, w_scale=cfg.data.w_scale,
        n_mels=cfg.data.n_mels, fps=cfg.data.fps)

    print(f'train={len(train_ds)} pieces  val={len(val_ds)} pieces', flush=True)

    # Spectrogram normalisation — computed from training set, stored in model params
    print('Computing spectrogram stats...', flush=True)
    means, stds = train_ds.compute_spec_stats()
    print(f'  means={means.mean():.4f}  stds={stds.mean():.4f}', flush=True)

    net_config = OmegaConf.to_container(cfg.net)
    network    = ConditionalUNet(net_config)
    network.perf_encoder.set_stats(means, stds)   # BEFORE .to(device)
    network    = network.to(device)
    print(f'params: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}',
          flush=True)

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=cfg.optim.lr,
                                 weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.optim.patience, factor=0.5)

    max_epochs     = cfg.train.max_epochs
    early_patience = cfg.optim.patience * 2   # 10 by default
    best_val_loss  = float('inf')
    wait           = 0
    best_path      = None
    t0             = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = _train_epoch(
            network, train_ds, optimizer,
            seq_len=cfg.train.seq_len,
            n_frames=cfg.data.n_frames,
            gt_width=cfg.data.gt_width,
            device=device)

        val_loss = _val_epoch(
            network, val_ds,
            seq_len=cfg.train.seq_len,
            n_frames=cfg.data.n_frames,
            gt_width=cfg.data.gt_width,
            device=device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(
            f'epoch {epoch:3d}/{max_epochs}  '
            f'train_dice={train_loss:.4f}  acc={train_acc:.3f}  '
            f'val_dice={val_loss:.4f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
            f'{elapsed:.0f}s',
            flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait          = 0
            best_path     = _save_best(network, epoch, cfg, out_dir)
            print(f'  -> new best (val={best_val_loss:.5f}), saved {best_path}', flush=True)
        else:
            wait += 1
            print(f'  no improvement ({wait}/{early_patience})', flush=True)
            if wait >= early_patience:
                print(f'Early stopping at epoch {epoch}.', flush=True)
                break

    print(f'Training done. best_val_loss={best_val_loss:.5f}  best={best_path}', flush=True)


def _parse() -> DictConfig:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/v11_cpjku_fullstrip.yaml')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg


if __name__ == '__main__':
    main(_parse())
