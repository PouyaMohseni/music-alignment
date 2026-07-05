"""CADP M04 training — dense tokens (no audio pooling), DTW-decoded at eval.

Same decode strategy as M01/M03 (whole-piece monotonic DTW) so this ablation
isolates the effect of dense audio+score tokens alone, separate from M05's
learned-decode fix. Expect the same DTW cascade-failure mode on tempo-drift/
repeat pieces that hit M01/M03 — that's the point of the comparison.

    python -m mymodel.cadp.m04_train --config configs/cadp_m04.yaml
"""
from __future__ import annotations
import argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from mymodel.cadp.m04_model import M04DenseTokens, subcol_positions
from mymodel.cadp.dataset import CADPDataset
from mymodel.shared.losses import expected_distance_loss

# Positions are raw pixels (thousands), but grad clipping below is tuned for
# M01/M03's normalized-[0,1] loss scale. Without this, pre-clip gradient norms
# run 500-6000 and get crushed to norm=1 every step, which silently prevents
# any real learning (confirmed empirically: train loss never decreases).
# Scaling positions only inside the loss call — not model.forward's own
# pred_pos, which stays in real pixels — fixes this without touching the model.
PIX_SCALE = 1000.0


def _build_training_sample(piece: dict, fps: float, win_sec: float, device: str):
    """Random win_sec audio window, dense (no pooling); full score always used
    (no attention here — just a matmul sim, so no O(N^2) memory blowup)."""
    mert = piece['mert_feats']    # (T, 768)
    d2   = piece['d2_feats']      # (N_cols, 16, 768)
    T = mert.shape[0]
    N_cols = d2.shape[0]

    win_frames = int(win_sec * fps)
    t_start = random.randint(0, max(0, T - win_frames))
    t_end   = min(t_start + win_frames, T)
    audio_win = mert[t_start:t_end]                          # (win_T, 768)

    audio_t = torch.from_numpy(audio_win).to(device)
    score_t = torch.from_numpy(d2).to(device)

    f_idx = piece['frame_idx']
    strip_x = piece['strip_x']
    in_win = (f_idx >= t_start) & (f_idx < t_end)

    win_T = t_end - t_start
    pos_target = torch.zeros(win_T, device=device)
    valid_mask = torch.zeros(win_T, dtype=torch.bool, device=device)
    if in_win.any():
        local_idx = (f_idx[in_win] - t_start).astype(np.int64)
        pos_target[local_idx] = torch.from_numpy(strip_x[in_win]).float().to(device)
        valid_mask[local_idx] = True

    return audio_t, score_t, pos_target, valid_mask, N_cols


def train(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = M04DenseTokens(
        mert_dim=cfg.model.mert_dim,
        dinov2_dim=cfg.model.dinov2_dim,
        hidden_dim=cfg.model.hidden_dim,
        embed_dim=cfg.model.embed_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')

    train_ds = CADPDataset(
        cfg.data.processed_root, cfg.data.mert_root, cfg.data.dinov2_root,
        split='train', fps=cfg.data.fps)
    val_ds = CADPDataset(
        cfg.data.processed_root, cfg.data.mert_root, cfg.data.dinov2_root,
        split='val', fps=cfg.data.fps)

    print('Loading train pieces...')
    train_pieces = [p for pid in train_ds.piece_ids
                    if (p := train_ds.load_piece(pid)) is not None]
    print(f'  {len(train_pieces)} train pieces loaded')
    print('Loading val pieces...')
    val_pieces = [p for pid in val_ds.piece_ids
                  if (p := val_ds.load_piece(pid)) is not None]
    print(f'  {len(val_pieces)} val pieces loaded')

    opt = optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                      weight_decay=cfg.optim.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=cfg.optim.patience, factor=0.5)

    col_stride = float(train_ds.col_stride)
    col_w = float(train_ds.col_w)

    best_val = float('inf')
    wait = 0

    for epoch in range(1, cfg.train.max_epochs + 1):
        model.train()
        random.shuffle(train_pieces)
        losses = []
        for piece in train_pieces:
            audio_t, score_t, pos_target, valid_mask, n_cols = _build_training_sample(
                piece, cfg.data.fps, cfg.train.win_sec, device)
            pos_subcol = subcol_positions(n_cols, col_stride, col_w).to(device)

            out = model(audio_t, score_t)
            loss, _ = expected_distance_loss(
                out['sim'], pos_subcol / PIX_SCALE, pos_target / PIX_SCALE, valid_mask,
                temperature=cfg.loss.temperature, power=cfg.loss.power,
                entropy_weight=cfg.loss.entropy_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else float('nan')

        model.eval()
        val_losses = []
        with torch.no_grad():
            for piece in val_pieces:
                audio_t, score_t, pos_target, valid_mask, n_cols = _build_training_sample(
                    piece, cfg.data.fps, cfg.train.win_sec, device)
                pos_subcol = subcol_positions(n_cols, col_stride, col_w).to(device)
                out = model(audio_t, score_t)
                loss, _ = expected_distance_loss(
                    out['sim'], pos_subcol / PIX_SCALE, pos_target / PIX_SCALE, valid_mask,
                    temperature=cfg.loss.temperature, power=cfg.loss.power)
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
        # loss is in units of PIX_SCALE px (scaled for gradient health, see above)
        print(f'Epoch {epoch:03d}{marker}  train={train_loss*PIX_SCALE:.1f}px  val={val_loss*PIX_SCALE:.1f}px'
              f'  best={best_val*PIX_SCALE:.1f}px  wait={wait}/{cfg.optim.patience * 2}',
              flush=True)

        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_loss': val_loss},
                       str(out_dir / f'checkpoint_epoch{epoch:03d}.pt'))

        if wait >= cfg.optim.patience * 2:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'Training done. Best val_loss={best_val*PIX_SCALE:.1f}px')


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
