"""CADP M01 training — frozen DINOv2 + MERT, sim matrix, expected_distance loss.

    python -m mymodel.cadp.m01_train --config configs/cadp_m01.yaml
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from mymodel.cadp.m01_model import M01FrozenBaseline
from mymodel.cadp.dataset import CADPDataset
from mymodel.shared.losses import expected_distance_loss


def _build_training_sample(piece: dict, n_chunks: int, fps: float,
                            win_sec: float = 5.0, device: str = 'cpu'):
    """Sample a random 5s window and return (audio_tensor, score_tensor, pos, mask)."""
    mert  = piece['mert_feats']    # (T, 768)
    d2    = piece['d2_feats']      # (N_cols, 16, 768)
    T = mert.shape[0]
    N = d2.shape[0]

    # Random 5s window
    win_frames = int(win_sec * fps)
    t_start = random.randint(0, max(0, T - win_frames))
    t_end   = min(t_start + win_frames, T)
    audio_win = mert[t_start:t_end]   # (win_T, 768)

    audio_t = torch.from_numpy(audio_win).unsqueeze(0).to(device)  # (1, win_T, 768)

    # Find notes in this window
    f_idx = piece['frame_idx']
    c_idx = piece['col_idx']
    in_win = (f_idx >= t_start) & (f_idx < t_end)

    # For each of the n_chunks audio slots, find the GT column position
    # (normalized to [0, 1] for expected_distance_loss)
    chunk_size = max(1, (t_end - t_start) // n_chunks)
    pos_target = torch.zeros(n_chunks, device=device)
    valid_mask = torch.zeros(n_chunks, dtype=torch.bool, device=device)

    for chunk_i in range(n_chunks):
        c_t0 = t_start + chunk_i * chunk_size
        c_t1 = c_t0 + chunk_size
        in_chunk = (f_idx >= c_t0) & (f_idx < c_t1) & in_win
        if in_chunk.any():
            mean_col = float(c_idx[in_chunk].mean())
            pos_target[chunk_i] = mean_col / max(N - 1, 1)
            valid_mask[chunk_i] = True

    # pos_tile: column centers normalized to [0, 1]
    pos_tile = torch.arange(N, dtype=torch.float32, device=device) / max(N - 1, 1)

    score_t = torch.from_numpy(d2).to(device)  # (N_cols, 16, 768)

    return audio_t, score_t, pos_tile, pos_target, valid_mask


def train(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = M01FrozenBaseline(
        mert_dim=cfg.model.mert_dim,
        dinov2_dim=cfg.model.dinov2_dim,
        embed_dim=cfg.model.embed_dim,
        n_audio_chunks=cfg.model.n_audio_chunks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')

    train_ds = CADPDataset(
        cfg.data.processed_root, cfg.data.mert_root, cfg.data.dinov2_root,
        split='train', fps=cfg.data.fps)
    val_ds = CADPDataset(
        cfg.data.processed_root, cfg.data.mert_root, cfg.data.dinov2_root,
        split='val', fps=cfg.data.fps)

    print(f'Loading train pieces...')
    train_pieces = [p for pid in train_ds.piece_ids
                    if (p := train_ds.load_piece(pid)) is not None]
    print(f'  {len(train_pieces)} train pieces loaded')
    print(f'Loading val pieces...')
    val_pieces = [p for pid in val_ds.piece_ids
                  if (p := val_ds.load_piece(pid)) is not None]
    print(f'  {len(val_pieces)} val pieces loaded')

    opt = optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                      weight_decay=cfg.optim.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=cfg.optim.patience, factor=0.5)

    best_val = float('inf')
    wait = 0
    n_chunks = cfg.model.n_audio_chunks

    # Diagnostic (scripts/debug_m01_*.py): stepping the optimizer after every
    # single piece (effective batch size 1) means each step's gradient points
    # toward a completely different randomly-resampled window/target, and
    # that noise swamps the learning signal once real dataset diversity (354
    # pieces) is added -- confirmed by fixed-window overfit working fine
    # (up to 50 pieces) while resampled-window training flatlines even at 5
    # pieces. Gradient accumulation over batch_size pieces per opt.step()
    # measurably recovers real loss reduction in that diagnostic.
    batch_size = getattr(cfg.train, 'batch_size', 32)

    for epoch in range(1, cfg.train.max_epochs + 1):
        model.train()
        random.shuffle(train_pieces)
        losses = []
        opt.zero_grad()
        for i, piece in enumerate(train_pieces):
            audio_t, score_t, pos_tile, pos_target, valid_mask = \
                _build_training_sample(piece, n_chunks, cfg.data.fps,
                                       cfg.train.win_sec, device)
            out = model(audio_t, score_t)
            sim = out['sim'].squeeze(0)   # (K, N_cols)
            loss, _ = expected_distance_loss(
                sim, pos_tile, pos_target, valid_mask,
                temperature=cfg.loss.temperature,
                power=cfg.loss.power,
                entropy_weight=cfg.loss.entropy_weight)
            (loss / batch_size).backward()
            losses.append(loss.item())
            if (i + 1) % batch_size == 0 or i == len(train_pieces) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        train_loss = float(np.mean(losses)) if losses else float('nan')

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for piece in val_pieces:
                audio_t, score_t, pos_tile, pos_target, valid_mask = \
                    _build_training_sample(piece, n_chunks, cfg.data.fps,
                                           cfg.train.win_sec, device)
                out = model(audio_t, score_t)
                sim = out['sim'].squeeze(0)
                loss, _ = expected_distance_loss(
                    sim, pos_tile, pos_target, valid_mask,
                    temperature=cfg.loss.temperature)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        sched.step(val_loss)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            wait = 0
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_loss': val_loss},
                       str(out_dir / 'best_model.pt'))
        else:
            wait += 1

        marker = '*' if improved else ' '
        print(f'Epoch {epoch:03d}{marker}  train={train_loss:.4f}  val={val_loss:.4f}'
              f'  best={best_val:.4f}  wait={wait}/{cfg.optim.patience * 2}',
              flush=True)

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_loss': val_loss},
                       str(out_dir / f'checkpoint_epoch{epoch:03d}.pt'))

        if wait >= cfg.optim.patience * 2:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'Training done. Best val_loss={best_val:.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('overrides', nargs='*')
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    train(cfg)


if __name__ == '__main__':
    main()
