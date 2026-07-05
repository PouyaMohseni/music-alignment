"""CADP M05 training — learned path predictor, direct pixel-position regression.

No DTW anywhere in this loop: supervision is expected_distance_loss between
the model's per-frame soft-argmax pixel position and the ground-truth strip_x
at annotated onset frames.

The axial self-attention over score positions costs O(T_a * N_s^2); the median
training piece has N_s (=N_cols*4) in the high hundreds and some reach several
thousand, which OOMs a 40GB A100 within the first backward pass. So score
columns are cropped to MAX_N_COLS, centered on the onsets that fall in the
current audio window (with margin), whenever a piece is wider than that cap.
Positions stay in absolute pixel space via subcol_positions' col_offset, so
this is invisible to the loss — it only shrinks how much of the strip a given
training step can see, not the coordinate system.

    python -m mymodel.cadp.m05_train --config configs/cadp_m05.yaml
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from mymodel.cadp.m05_model import M05LearnedPathPredictor
from mymodel.cadp.m04_model import subcol_positions
from mymodel.cadp.dataset import CADPDataset
from mymodel.shared.losses import expected_distance_loss

# Positions are raw pixels (thousands), but grad clipping below is tuned for
# M01/M03's normalized-[0,1] loss scale. Without this, pre-clip gradient norms
# run 500-6000 and get crushed to norm=1 every step, which silently prevents
# any real learning (confirmed empirically: train loss never decreases).
# Scaling positions only inside the loss call — not model.forward's own
# pred_pos, which stays in real pixels — fixes this without touching the model.
PIX_SCALE = 1000.0


def _crop_score(d2: np.ndarray, col_idx: np.ndarray, onset_cols_in_win: np.ndarray,
                 max_n_cols: int) -> tuple[np.ndarray, int]:
    """Crop d2 (N_cols,16,768) to at most max_n_cols columns, centered on the
    onsets in the current audio window (random center if none). Returns
    (cropped_d2, col_offset) — col_offset is the absolute start column so
    subcol_positions can stay in global pixel space.
    """
    n_cols = d2.shape[0]
    if n_cols <= max_n_cols:
        return d2, 0
    if len(onset_cols_in_win) > 0:
        center = int(round(float(onset_cols_in_win.mean())))
    else:
        center = random.randint(0, n_cols - 1)
    start = max(0, min(n_cols - max_n_cols, center - max_n_cols // 2))
    return d2[start:start + max_n_cols], start


def _build_training_sample(piece: dict, fps: float, win_sec: float, device: str,
                            max_n_cols: int):
    """Random win_sec audio window; score cropped to max_n_cols if the piece
    is wider (see module docstring — avoids O(T_a * N_s^2) attention OOM)."""
    mert = piece['mert_feats']    # (T, 768)
    d2   = piece['d2_feats']      # (N_cols, 16, 768)
    T = mert.shape[0]

    win_frames = int(win_sec * fps)
    t_start = random.randint(0, max(0, T - win_frames))
    t_end   = min(t_start + win_frames, T)
    audio_win = mert[t_start:t_end]                          # (win_T, 768)

    f_idx = piece['frame_idx']
    c_idx = piece['col_idx']
    strip_x = piece['strip_x']
    in_win = (f_idx >= t_start) & (f_idx < t_end)

    d2_crop, col_offset = _crop_score(d2, c_idx, c_idx[in_win], max_n_cols)

    audio_t = torch.from_numpy(audio_win).to(device)          # (win_T, 768)
    score_t = torch.from_numpy(d2_crop).to(device)              # (n_cols_crop, 16, 768)

    win_T = t_end - t_start
    pos_target = torch.zeros(win_T, device=device)
    valid_mask = torch.zeros(win_T, dtype=torch.bool, device=device)
    if in_win.any():
        local_idx = (f_idx[in_win] - t_start).astype(np.int64)
        pos_target[local_idx] = torch.from_numpy(strip_x[in_win]).float().to(device)
        valid_mask[local_idx] = True

    return audio_t, score_t, pos_target, valid_mask, d2_crop.shape[0], col_offset


def train(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = M05LearnedPathPredictor(
        mert_dim=cfg.model.mert_dim,
        dinov2_dim=cfg.model.dinov2_dim,
        hidden_dim=cfg.model.hidden_dim,
        embed_dim=cfg.model.embed_dim,
        path_channels=cfg.model.path_channels,
        attention_heads=cfg.model.attention_heads,
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
            audio_t, score_t, pos_target, valid_mask, n_cols, col_offset = _build_training_sample(
                piece, cfg.data.fps, cfg.train.win_sec, device, cfg.model.max_n_cols)
            pos_subcol = subcol_positions(n_cols, col_stride, col_w, col_offset).to(device)

            out = model(audio_t, score_t, pos_subcol, temperature=cfg.loss.temperature)
            # Same temperature as the model's own softmax, so expected_distance_loss's
            # internal softmax(logits/temperature) reproduces out['p'] exactly.
            loss, _ = expected_distance_loss(
                out['logits'], pos_subcol / PIX_SCALE, pos_target / PIX_SCALE, valid_mask,
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
                audio_t, score_t, pos_target, valid_mask, n_cols, col_offset = _build_training_sample(
                    piece, cfg.data.fps, cfg.train.win_sec, device, cfg.model.max_n_cols)
                pos_subcol = subcol_positions(n_cols, col_stride, col_w, col_offset).to(device)
                out = model(audio_t, score_t, pos_subcol, temperature=cfg.loss.temperature)
                loss, _ = expected_distance_loss(
                    out['logits'], pos_subcol / PIX_SCALE, pos_target / PIX_SCALE, valid_mask,
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
