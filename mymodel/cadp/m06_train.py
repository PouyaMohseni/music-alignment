"""CADP M06 training — INR head, continuous-position heatmap regression.

Same score-cropping rationale as M05 (axial attention is O(T_a*N_s^2); see
that file's docstring) — crop to max_n_cols columns centered on the current
audio window's onsets. Within the crop, queries are placed on a dense
continuous linspace (query_resolution_multiplier x finer than the x4
sub-column grid M01-M05 used), which is what lets this model resolve
positions between column boundaries instead of snapping to them.

    python -m mymodel.cadp.m06_train --config configs/cadp_m06.yaml
"""
from __future__ import annotations
import argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from mymodel.cadp.m06_model import M06INRHead
from mymodel.cadp.m05_train import _crop_score
from mymodel.cadp.dataset import CADPDataset
from mymodel.shared.losses import heatmap_inr_loss


def _build_training_sample(piece: dict, fps: float, win_sec: float, device: str,
                            max_n_cols: int, col_stride: float, col_w: float,
                            query_res_mult: int):
    mert = piece['mert_feats']    # (T, 768)
    d2   = piece['d2_feats']      # (N_cols, 16, 768)
    T = mert.shape[0]

    win_frames = int(win_sec * fps)
    t_start = random.randint(0, max(0, T - win_frames))
    t_end   = min(t_start + win_frames, T)
    audio_win = mert[t_start:t_end]

    f_idx = piece['frame_idx']
    c_idx = piece['col_idx']
    strip_x = piece['strip_x']
    in_win = (f_idx >= t_start) & (f_idx < t_end)

    d2_crop, col_offset = _crop_score(d2, c_idx, c_idx[in_win], max_n_cols)
    n_cols_crop = d2_crop.shape[0]

    audio_t = torch.from_numpy(audio_win).to(device)
    score_t = torch.from_numpy(d2_crop).to(device)

    win_T = t_end - t_start
    pos_target = torch.zeros(win_T, device=device)
    valid_mask = torch.zeros(win_T, dtype=torch.bool, device=device)
    if in_win.any():
        local_idx = (f_idx[in_win] - t_start).astype(np.int64)
        pos_target[local_idx] = torch.from_numpy(strip_x[in_win]).float().to(device)
        valid_mask[local_idx] = True

    # Dense continuous query grid over the crop's pixel range.
    px_lo = col_offset * col_stride
    px_hi = (col_offset + n_cols_crop - 1) * col_stride + col_w
    n_queries = n_cols_crop * 4 * query_res_mult
    query_x = torch.linspace(px_lo, px_hi, n_queries, device=device)

    return audio_t, score_t, pos_target, valid_mask, query_x


def train(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = M06INRHead(
        mert_dim=cfg.model.mert_dim,
        dinov2_dim=cfg.model.dinov2_dim,
        hidden_dim=cfg.model.hidden_dim,
        embed_dim=cfg.model.embed_dim,
        path_channels=cfg.model.path_channels,
        attention_heads=cfg.model.attention_heads,
        cond_dim=cfg.model.cond_dim,
        inr_hidden_dim=cfg.model.inr_hidden_dim,
        fourier_freqs=tuple(cfg.model.fourier_freqs),
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
    max_n_cols = cfg.model.max_n_cols
    query_res_mult = cfg.model.query_resolution_multiplier
    sigma_px = cfg.loss.heatmap_sigma_px

    best_val = float('inf')
    wait = 0

    for epoch in range(1, cfg.train.max_epochs + 1):
        model.train()
        random.shuffle(train_pieces)
        losses = []
        for piece in train_pieces:
            audio_t, score_t, pos_target, valid_mask, query_x = _build_training_sample(
                piece, cfg.data.fps, cfg.train.win_sec, device,
                max_n_cols, col_stride, col_w, query_res_mult)

            out = model(audio_t, score_t, query_x)
            loss, _ = heatmap_inr_loss(
                out['confidence'], query_x, pos_target, valid_mask, sigma_px=sigma_px)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else float('nan')

        model.eval()
        val_losses = []
        val_px_errs = []
        with torch.no_grad():
            for piece in val_pieces:
                audio_t, score_t, pos_target, valid_mask, query_x = _build_training_sample(
                    piece, cfg.data.fps, cfg.train.win_sec, device,
                    max_n_cols, col_stride, col_w, query_res_mult)
                out = model(audio_t, score_t, query_x)
                loss, _ = heatmap_inr_loss(
                    out['confidence'], query_x, pos_target, valid_mask, sigma_px=sigma_px)
                val_losses.append(loss.item())
                if valid_mask.any():
                    err = (out['pred_pos'][valid_mask] - pos_target[valid_mask]).abs().mean()
                    val_px_errs.append(err.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        val_px = float(np.mean(val_px_errs)) if val_px_errs else float('nan')

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
        print(f'Epoch {epoch:03d}{marker}  train_ce={train_loss:.3f}  val_ce={val_loss:.3f}'
              f'  val_px={val_px:.1f}px  best={best_val:.3f}  wait={wait}/{cfg.optim.patience * 2}',
              flush=True)

        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_loss': val_loss},
                       str(out_dir / f'checkpoint_epoch{epoch:03d}.pt'))

        if wait >= cfg.optim.patience * 2:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'Training done. Best val_ce={best_val:.3f}')


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
