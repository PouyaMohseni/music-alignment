"""C1: BPTT training for the audio-visual grounding cross-attention model.

Mirrors mymodel/v11_cpjku_fullstrip/train.py's structure exactly (same
FullStripDataset, same BPTT chunking, same dice loss, same
checkpoint/resume/early-stop machinery) -- only the network differs
(C1VisualGroundingNet instead of ConditionalUNet), since the architecture
change is deep enough that monkey-patching CPJKU's train_model.py (the
pattern used for B1-B6) isn't applicable here.

    python -m mymodel.c1_visual_grounding.train --config configs/c1_visual_grounding.yaml
"""
from __future__ import annotations
import argparse, os, random, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mymodel.v11_cpjku_fullstrip.data import FullStripDataset, make_gt_mask
from .model import C1VisualGroundingNet

EPS = 1e-8


def _seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smoothing) /
                (iflat.pow(2).sum() + tflat.pow(2).sum() + smoothing + EPS))


def _build_perf(spec: np.ndarray, frames: list[int], n_frames: int) -> np.ndarray:
    n_mels = spec.shape[0]
    sl = len(frames)
    out = np.zeros((sl, 1, n_mels, n_frames), dtype=np.float32)
    for i, t in enumerate(frames):
        t0 = max(0, t - n_frames)
        window = spec[:, t0:t]
        if window.shape[-1] < n_frames:
            window = np.pad(window, ((0, 0), (n_frames - window.shape[-1], 0)))
        out[i, 0] = window
    return out


def _build_gt(H: int, W_sc: int, strip_x_sc: np.ndarray, frames: list[int], gt_width: int) -> np.ndarray:
    sl = len(frames)
    out = np.zeros((sl, 1, H, W_sc), dtype=np.float32)
    for i, t in enumerate(frames):
        cx = int(np.clip(round(strip_x_sc[t]), 0, W_sc - 1))
        out[i, 0] = make_gt_mask(H, W_sc, cx=cx, gt_width=gt_width)
    return out


def _init_hidden(network: C1VisualGroundingNet, device: str):
    return (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
            torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))


def _detach_hidden(hidden):
    if hidden is None:
        return None
    return tuple(h.detach() for h in hidden)


def _forward_chunk(network, score_1frame, spec, strip_x_sc, frames, n_frames, gt_width, hidden, device):
    sl = len(frames)
    H = score_1frame.shape[-2]
    W_sc = score_1frame.shape[-1]

    score_batch = score_1frame.expand(sl, -1, -1, -1, -1)

    perf_np = _build_perf(spec, frames, n_frames)
    perf_batch = torch.from_numpy(perf_np).to(device).unsqueeze(1)

    gt_np = _build_gt(H, W_sc, strip_x_sc, frames, gt_width)
    gt_batch = torch.from_numpy(gt_np).to(device)

    out = network(score=score_batch, perf=perf_batch, hidden=hidden)
    pred = out['segmentation']
    new_hidden = _detach_hidden(out.get('hidden'))

    loss = dice_loss(pred, gt_batch, smoothing=0.0)

    with torch.no_grad():
        col = pred.squeeze(1).sum(dim=1)
        pred_x = col.argmax(dim=-1).float()
        gt_x = torch.tensor([strip_x_sc[f] for f in frames], device=device, dtype=torch.float32)
        acc = ((pred_x - gt_x).abs() <= W_sc * 0.1).float().mean()

    return loss, float(acc), new_hidden


def _train_epoch(network, dataset, optimizer, seq_len, n_frames, gt_width, device):
    network.train()
    order = list(range(len(dataset)))
    np.random.shuffle(order)

    total_loss = total_acc = n_frames_seen = 0

    for idx in order:
        piece = dataset[idx]
        score, spec, strip_x_sc, T = piece['score'], piece['spec'], piece['strip_x_sc'], piece['T']

        score_1 = torch.from_numpy(score[np.newaxis, np.newaxis, np.newaxis]).to(device, non_blocking=True)
        hidden = _init_hidden(network, device)
        t = 0
        while t < T:
            frames = list(range(t, min(t + seq_len, T)))
            sl = len(frames)

            optimizer.zero_grad(set_to_none=True)
            loss, acc, hidden = _forward_chunk(network, score_1, spec, strip_x_sc, frames, n_frames, gt_width, hidden, device)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * sl
            total_acc += acc * sl
            n_frames_seen += sl
            t += sl

    denom = max(1, n_frames_seen)
    return total_loss / denom, total_acc / denom


@torch.no_grad()
def _val_epoch(network, dataset, seq_len, n_frames, gt_width, device):
    network.eval()
    total_loss = n_frames_seen = 0
    for idx in range(len(dataset)):
        piece = dataset[idx]
        score, spec, strip_x_sc, T = piece['score'], piece['spec'], piece['strip_x_sc'], piece['T']
        score_1 = torch.from_numpy(score[np.newaxis, np.newaxis, np.newaxis]).to(device, non_blocking=True)
        hidden = _init_hidden(network, device)
        t = 0
        while t < T:
            frames = list(range(t, min(t + seq_len, T)))
            sl = len(frames)
            loss, _, hidden = _forward_chunk(network, score_1, spec, strip_x_sc, frames, n_frames, gt_width, hidden, device)
            total_loss += float(loss) * sl
            n_frames_seen += sl
            t += sl
    network.train()
    return total_loss / max(1, n_frames_seen)


def _save_checkpoint(network, optimizer, scheduler, epoch, best_val_loss, wait,
                     means, stds, cfg, out_dir, is_best):
    ckpt_path = out_dir / f'checkpoint_epoch{epoch:03d}.pt'
    best_path = out_dir / 'best_model.pt'
    payload = {
        'epoch': epoch, 'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'best_val_loss': best_val_loss, 'wait': wait,
        'spec_means': means, 'spec_stds': stds,
        'net_config': OmegaConf.to_container(cfg.net),
        'cfg': OmegaConf.to_container(cfg),
    }
    torch.save(payload, ckpt_path)
    if is_best:
        torch.save(payload, best_path)
    old = out_dir / f'checkpoint_epoch{epoch - 2:03d}.pt'
    if old.exists():
        old.unlink()
    return ckpt_path


def main(cfg: DictConfig, resume: str | None = None):
    _seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(os.getcwd()) / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}  out={out_dir}', flush=True)

    train_ds = FullStripDataset(cfg.data.processed_root, 'train', h_strip=cfg.data.h_strip,
                                w_scale=cfg.data.w_scale, n_mels=cfg.data.n_mels, fps=cfg.data.fps)
    val_ds = FullStripDataset(cfg.data.processed_root, 'val', h_strip=cfg.data.h_strip,
                              w_scale=cfg.data.w_scale, n_mels=cfg.data.n_mels, fps=cfg.data.fps)
    print(f'train={len(train_ds)} pieces  val={len(val_ds)} pieces', flush=True)

    network = C1VisualGroundingNet(
        spec_enc=cfg.net.spec_enc, rnn_size=cfg.net.rnn_size, rnn_layers=cfg.net.rnn_layer,
        d_model=cfg.net.d_model, n_heads=cfg.net.n_heads, patch_w=cfg.net.patch_w)

    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=cfg.optim.patience, factor=0.5)

    max_epochs = cfg.train.max_epochs
    early_patience = cfg.optim.patience * 2
    best_val_loss = float('inf')
    wait = 0
    start_epoch = 1
    best_path = None

    ckpt_to_load = None
    if resume:
        ckpt_to_load = resume if resume != 'auto' else None
        if ckpt_to_load is None:
            candidates = sorted(out_dir.glob('checkpoint_epoch*.pt'))
            if candidates:
                ckpt_to_load = str(candidates[-1])
                print(f'Auto-resume: found {ckpt_to_load}', flush=True)
            else:
                print('Auto-resume: no checkpoint found, starting fresh.', flush=True)

    if ckpt_to_load and Path(ckpt_to_load).exists():
        print(f'Resuming from {ckpt_to_load}', flush=True)
        ckpt = torch.load(ckpt_to_load, map_location='cpu', weights_only=False)
        network.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_val_loss = ckpt['best_val_loss']
        wait = ckpt['wait']
        start_epoch = ckpt['epoch'] + 1
        means = ckpt['spec_means']
        stds = ckpt['spec_stds']
        print(f'  resumed epoch={ckpt["epoch"]}  best_val={best_val_loss:.5f}  wait={wait}', flush=True)
    else:
        print('Computing spectrogram stats...', flush=True)
        means, stds = train_ds.compute_spec_stats()
        print(f'  means={means.mean():.4f}  stds={stds.mean():.4f}', flush=True)

    network.perf_encoder.set_stats(means, stds)
    network = network.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    print(f'params: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}', flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        train_loss, train_acc = _train_epoch(network, train_ds, optimizer, seq_len=cfg.train.seq_len,
                                             n_frames=cfg.data.n_frames, gt_width=cfg.data.gt_width, device=device)
        val_loss = _val_epoch(network, val_ds, seq_len=cfg.train.seq_len,
                              n_frames=cfg.data.n_frames, gt_width=cfg.data.gt_width, device=device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        is_best = val_loss < best_val_loss
        print(f'epoch {epoch:3d}/{max_epochs}  train_dice={train_loss:.4f}  acc={train_acc:.3f}  '
              f'val_dice={val_loss:.4f}  lr={optimizer.param_groups[0]["lr"]:.2e}  {elapsed:.0f}s', flush=True)

        if is_best:
            best_val_loss = val_loss
            wait = 0
            best_path = _save_checkpoint(network, optimizer, scheduler, epoch, best_val_loss, wait,
                                         means, stds, cfg, out_dir, is_best=True)
            print(f'  -> new best (val={best_val_loss:.5f}), saved {best_path}', flush=True)
        else:
            wait += 1
            _save_checkpoint(network, optimizer, scheduler, epoch, best_val_loss, wait,
                             means, stds, cfg, out_dir, is_best=False)
            print(f'  no improvement ({wait}/{early_patience})', flush=True)
            if wait >= early_patience:
                print(f'Early stopping at epoch {epoch}.', flush=True)
                break

    print(f'Training done. best_val_loss={best_val_loss:.5f}  best={best_path}', flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/c1_visual_grounding.yaml')
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
