"""v11-mert-finetune: BPTT training matching v11_cpjku_fullstrip, but with a
LIVE, fine-tunable MERT-v1-95M (mymodel/v10_mert_unet/mert_live.py) as the
audio encoder instead of a static spectrogram. Every other MERT usage in
this project (v13/v14/v15, B1a) reads precomputed, FROZEN embeddings --
MERT's self-supervised pretraining was never optimized for this task's
50ms-precision localization, and a frozen readout can't adapt that. This is
the first experiment where MERT's own weights receive gradient from the
alignment task.

Compute trade-off: MERT (95M params) is far more expensive per call than
CBEncoder's small CNN, so unlike v11's seq_len=16 (0.8s) BPTT chunks, this
uses a longer seq_len (default 64, 3.2s) to amortize one MERT forward+backward
call over more frames -- fewer, larger MERT calls per piece. window_sec
controls how much left-context (beyond the chunk itself) MERT sees per call;
consecutive chunks' windows overlap and MERT recomputes embeddings for
overlapping audio (wasteful but correct -- there's no cheap way to carry
MERT's own attention state across chunks the way the LSTM's hidden state is
carried).

    python -m mymodel.v11_mert_finetune.train --config configs/v11_mert_finetune.yaml
"""
from __future__ import annotations
import argparse, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v10_mert_unet.mert_live import MERTLive, MERT_SR
from .data import FullStripAudioDataset, load_raw_audio

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


def _make_gt_mask(H: int, W: int, cx: int, gt_width: int = 10) -> np.ndarray:
    gt_height = H // 2
    mask = np.zeros((H, W), dtype=np.float32)
    cy = H // 2
    y0 = max(0, cy - gt_height // 2)
    y1 = min(H, cy + gt_height // 2)
    x0 = max(0, cx - gt_width // 2)
    x1 = min(W, cx + gt_width // 2)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def _build_gt(H: int, W_sc: int, strip_x_sc: np.ndarray,
              frames: list[int], gt_width: int) -> np.ndarray:
    sl = len(frames)
    out = np.zeros((sl, 1, H, W_sc), dtype=np.float32)
    for i, t in enumerate(frames):
        cx = int(np.clip(round(strip_x_sc[t]), 0, W_sc - 1))
        out[i, 0] = _make_gt_mask(H, W_sc, cx=cx, gt_width=gt_width)
    return out


def _build_perf_mert(mert_live: MERTLive, audio_24k: torch.Tensor,
                     frames: list[int], fps: int, window_sec: float,
                     device: str) -> torch.Tensor:
    """Live MERT embeddings for one BPTT chunk. Returns (sl, 1, 1, 768, 1),
    matching MERTProjector's expected (seq_len, bs, c, 768, n_input_frames=1)."""
    sl = len(frames)
    win_samples = int(window_sec * MERT_SR)
    end_sample = min(audio_24k.shape[0], int(round((frames[-1] + 1) / fps * MERT_SR)))
    start_sample = max(0, end_sample - win_samples)
    window = audio_24k[start_sample:end_sample]
    if window.shape[0] < win_samples:
        window = F.pad(window, (win_samples - window.shape[0], 0))
    emb = mert_live.embed_window(window, n_frames_20fps=sl)   # (sl, 768)
    return emb.view(sl, 1, 1, 768, 1)


def _init_hidden(network: ConditionalUNet, device: str):
    if not network.use_lstm:
        return None
    return (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
            torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))


def _detach_hidden(hidden):
    if hidden is None:
        return None
    return tuple(h.detach() for h in hidden)


def _forward_chunk(network, mert_live, score_1frame, audio_24k, strip_x_sc,
                   frames, fps, window_sec, gt_width, hidden, device):
    sl = len(frames)
    H  = score_1frame.shape[-2]
    W_sc = score_1frame.shape[-1]

    score_batch = score_1frame.expand(sl, -1, -1, -1, -1)

    perf_batch = _build_perf_mert(mert_live, audio_24k, frames, fps, window_sec, device)

    gt_np = _build_gt(H, W_sc, strip_x_sc, frames, gt_width)
    gt_batch = torch.from_numpy(gt_np).to(device)

    out  = network(score=score_batch, perf=perf_batch, hidden=hidden)
    pred = out['segmentation']
    new_hidden = _detach_hidden(out.get('hidden'))

    loss = dice_loss(pred, gt_batch, smoothing=0.0)

    with torch.no_grad():
        col    = pred.squeeze(1).sum(dim=1)
        pred_x = col.argmax(dim=-1).float()
        gt_x   = torch.tensor([strip_x_sc[f] for f in frames],
                              device=device, dtype=torch.float32)
        acc    = ((pred_x - gt_x).abs() <= W_sc * 0.1).float().mean()

    return loss, float(acc), new_hidden


def _train_epoch(network, mert_live, dataset: FullStripAudioDataset,
                 optimizer: torch.optim.Optimizer,
                 seq_len: int, window_sec: float, gt_width: int,
                 fps: int, device: str) -> tuple[float, float]:
    network.train()
    mert_live.train()
    order = list(range(len(dataset)))
    np.random.shuffle(order)

    total_loss = total_acc = n_frames_seen = 0

    for idx in order:
        piece      = dataset[idx]
        score      = piece['score']
        audio_24k  = torch.from_numpy(piece['audio']).to(device, non_blocking=True)
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

            optimizer.zero_grad(set_to_none=True)
            loss, acc, hidden = _forward_chunk(
                network, mert_live, score_1, audio_24k, strip_x_sc,
                frames, fps, window_sec, gt_width, hidden, device)

            loss.backward()
            optimizer.step()

            total_loss    += float(loss.detach()) * sl
            total_acc     += acc * sl
            n_frames_seen += sl
            t             += sl

    denom = max(1, n_frames_seen)
    return total_loss / denom, total_acc / denom


@torch.no_grad()
def _val_epoch(network, mert_live, dataset: FullStripAudioDataset,
               seq_len: int, window_sec: float, gt_width: int,
               fps: int, device: str) -> float:
    network.eval()
    mert_live.eval()
    total_loss = n_frames_seen = 0

    for idx in range(len(dataset)):
        piece      = dataset[idx]
        score      = piece['score']
        audio_24k  = torch.from_numpy(piece['audio']).to(device, non_blocking=True)
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
                network, mert_live, score_1, audio_24k, strip_x_sc,
                frames, fps, window_sec, gt_width, hidden, device)

            total_loss    += float(loss) * sl
            n_frames_seen += sl
            t             += sl

    network.train()
    mert_live.train()
    return total_loss / max(1, n_frames_seen)


def _save_checkpoint(network, mert_live, optimizer, scheduler,
                     epoch: int, best_val_loss: float, wait: int,
                     cfg: DictConfig, out_dir: Path, is_best: bool) -> Path:
    ckpt_path = out_dir / f'checkpoint_epoch{epoch:03d}.pt'
    best_path  = out_dir / 'best_model.pt'
    payload = {
        'epoch':          epoch,
        'state_dict':     network.state_dict(),
        'mert_state_dict': mert_live.state_dict(),
        'optimizer':      optimizer.state_dict(),
        'scheduler':      scheduler.state_dict(),
        'best_val_loss':  best_val_loss,
        'wait':           wait,
        'net_config':     OmegaConf.to_container(cfg.net),
        'cfg':            OmegaConf.to_container(cfg),
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
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(os.getcwd()) / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}  out={out_dir}', flush=True)

    train_ds = FullStripAudioDataset(
        cfg.data.processed_root, 'train',
        h_strip=cfg.data.h_strip, w_scale=cfg.data.w_scale, fps=cfg.data.fps)

    val_ds = FullStripAudioDataset(
        cfg.data.processed_root, 'val',
        h_strip=cfg.data.h_strip, w_scale=cfg.data.w_scale, fps=cfg.data.fps)

    print(f'train={len(train_ds)} pieces  val={len(val_ds)} pieces', flush=True)

    net_config = OmegaConf.to_container(cfg.net)
    network    = ConditionalUNet(net_config)
    mert_live  = MERTLive(mert_id=cfg.mert.mert_id,
                          unfreeze_last_n=cfg.mert.get('unfreeze_last_n', None))

    trainable_params = (list(network.parameters()) + mert_live.trainable_parameters())
    optimizer = torch.optim.Adam(trainable_params,
                                 lr=cfg.optim.lr,
                                 weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.optim.patience, factor=0.5)

    max_epochs     = cfg.train.max_epochs
    early_patience = cfg.optim.patience * 2
    best_val_loss  = float('inf')
    wait           = 0
    start_epoch    = 1
    best_path      = None

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
        mert_live.load_state_dict(ckpt['mert_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_val_loss = ckpt['best_val_loss']
        wait          = ckpt['wait']
        start_epoch   = ckpt['epoch'] + 1
        print(f'  resumed epoch={ckpt["epoch"]}  best_val={best_val_loss:.5f}  '
              f'wait={wait}  lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True)

    network   = network.to(device)
    mert_live = mert_live.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    print(f'params (network): {sum(p.numel() for p in network.parameters() if p.requires_grad):,}',
          flush=True)
    print(f'params (mert, trainable): {sum(p.numel() for p in mert_live.trainable_parameters()):,}',
          flush=True)

    t0 = time.time()

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss, train_acc = _train_epoch(
            network, mert_live, train_ds, optimizer,
            seq_len=cfg.train.seq_len, window_sec=cfg.train.window_sec,
            gt_width=cfg.data.gt_width, fps=cfg.data.fps, device=device)

        val_loss = _val_epoch(
            network, mert_live, val_ds,
            seq_len=cfg.train.seq_len, window_sec=cfg.train.window_sec,
            gt_width=cfg.data.gt_width, fps=cfg.data.fps, device=device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        print(
            f'epoch {epoch:3d}/{max_epochs}  '
            f'train_dice={train_loss:.4f}  acc={train_acc:.3f}  '
            f'val_dice={val_loss:.4f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
            f'{elapsed:.0f}s',
            flush=True)

        if is_best:
            best_val_loss = val_loss
            wait          = 0
            best_path     = _save_checkpoint(
                network, mert_live, optimizer, scheduler, epoch,
                best_val_loss, wait, cfg, out_dir, is_best=True)
            print(f'  -> new best (val={best_val_loss:.5f}), saved {best_path}', flush=True)
        else:
            wait += 1
            _save_checkpoint(
                network, mert_live, optimizer, scheduler, epoch,
                best_val_loss, wait, cfg, out_dir, is_best=False)
            print(f'  no improvement ({wait}/{early_patience})', flush=True)
            if wait >= early_patience:
                print(f'Early stopping at epoch {epoch}.', flush=True)
                break

    print(f'Training done. best_val_loss={best_val_loss:.5f}  best={best_path}', flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/v11_mert_finetune.yaml')
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
