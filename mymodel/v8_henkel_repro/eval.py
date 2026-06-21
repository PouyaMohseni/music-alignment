"""v8 eval: faithful Henkel repro — tiled strip inference + DTW + sub-tile argmax.

Inference strategy (offline, compatible with existing henkel_metrics):
  1. audio.wav → CQT → audio CNN → LSTM → h[0..T-1]        (all context vectors)
  2. strip.png tiled into N windows of width tile_width       (50% overlap)
  3. Encode all tiles through U-Net encoder once (shared encoder features).
  4. For each audio frame t: decode all N tiles with FiLM(h_t) → (N, 1, W) pos maps
       → tile_score[n] = max of pos_map_n  (is tile n the right one?)
  5. (T, N) tile-score matrix → DTW → alignment path (which tile at each frame)
  6. Sub-tile position: pos_map_n[t].argmax() → local_x → strip_x = offset_n + local_x

    python -m mymodel.v8_henkel_repro.eval \
        --checkpoint results/v8_henkel_repro/checkpoint_030000.pt \
        --config configs/v8_henkel_repro.yaml --split test \
        --processed /project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .model import HenkelRepro, HenkelReproConfig, _EncOut
from .data import load_cqt, load_strip
from ..shared.metrics import alignment_metrics, henkel_metrics, dtw_backtrack


def _build(cfg, device):
    hc = HenkelReproConfig(
        n_bins=cfg.model.n_bins,
        cnn_channels=list(cfg.model.cnn_channels),
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        lstm_bidirectional=cfg.model.get("lstm_bidirectional", False),
        unet_channels=list(cfg.model.unet_channels),
        tile_width=cfg.model.tile_width)
    return HenkelRepro(hc).to(device)


def _tile_strip(strip_1d: np.ndarray, tile_width: int,
                stride: int) -> tuple[list[np.ndarray], list[int]]:
    """Divide 1-D strip into overlapping tiles of width tile_width.

    strip_1d : (1, W_full) float32
    Returns  : tiles (list of (1, tile_width) arrays), offsets (list of int)
    """
    W = strip_1d.shape[-1]
    tiles, offsets = [], []
    x = 0
    while x < W:
        x1 = min(x + tile_width, W)
        crop = strip_1d[:, x:x1]
        if crop.shape[-1] < tile_width:
            pad = np.zeros((1, tile_width - crop.shape[-1]), dtype=np.float32)
            crop = np.concatenate([crop, pad], axis=-1)
        tiles.append(crop)
        offsets.append(x)
        if x1 == W:
            break
        x += stride
    return tiles, offsets


@torch.no_grad()
def eval_split(checkpoint, cfg_path, processed_root, split,
               out_dir=None, limit=None, device=None, band=0.25,
               batch_tiles=64):
    cfg    = OmegaConf.load(cfg_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = _build(cfg, device)
    sd     = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"], strict=True)
    model.eval()

    hop  = cfg.data.hop
    sr   = cfg.data.sr
    W    = cfg.model.tile_width
    eff_hz = sr / hop

    proc = Path(processed_root)
    splits_data = json.load(open(proc / "splits.json"))
    piece_ids = splits_data[split][:limit] if limit else splits_data[split]

    out_root = Path(out_dir or str(Path(checkpoint).parent / "eval")) / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_root / "per_piece.jsonl", "w") as fout:
        for k, pid in enumerate(piece_ids):
            try:
                piece_dir = proc / pid
                ann   = json.load(open(piece_dir / "annotations.json"))
                notes = np.load(piece_dir / "noteheads.npz")
                strip_w    = ann["image"]["width_px"]
                dur        = float(ann["audio"]["duration_sec"])
                px_per_sec = strip_w / dur

                # ── Audio → context vectors h[0..T-1] ───────────────────
                cqt = load_cqt(piece_dir / "audio.wav",
                                sr=sr, n_bins=cfg.model.n_bins, hop=hop)  # (1, n_bins, T)
                T = cqt.shape[-1]
                cqt_t = cqt.unsqueeze(0).to(device)   # (1, 1, n_bins, T)
                h_all = model.audio_context(cqt_t)[0]  # (T, lstm_h)

                # ── Tile strip, encode each tile ──────────────────────────
                strip_full = load_strip(piece_dir / "strip.png")    # (1, H, W_full)
                strip_1d   = strip_full.mean(axis=1, keepdims=False) # (W_full,) then back to (1,W)
                strip_1d   = strip_full.mean(axis=1)[np.newaxis]     # (1, W_full)
                stride     = W // 2
                tiles, offsets = _tile_strip(strip_1d, W, stride)
                N = len(tiles)

                # Encode all tiles (U-Net encoder, once)
                tile_encs = []
                for ti in range(0, N, batch_tiles):
                    batch = torch.from_numpy(
                        np.stack([t[np.newaxis] for t in tiles[ti:ti+batch_tiles]])
                    ).to(device)  # (B, 1, 1, W) → need (B, 1, W)
                    batch = batch.squeeze(2)  # (B, 1, W)
                    enc = model.unet.encode(batch)
                    # Store as CPU tensors to save GPU memory across frames
                    tile_encs.append(_EncOut(
                        skips=[s.cpu() for s in enc.skips],
                        bottom=enc.bottom.cpu()))

                # Flatten tile_encs into one big EncOut
                all_skips = [
                    torch.cat([te.skips[i] for te in tile_encs], dim=0)
                    for i in range(len(tile_encs[0].skips))]
                all_bottom = torch.cat([te.bottom for te in tile_encs], dim=0)
                enc_all = _EncOut(skips=all_skips, bottom=all_bottom)

                # ── Build (T, N) tile-score matrix ────────────────────────
                tile_scores   = np.zeros((T, N), dtype=np.float32)
                sub_tile_argmax = np.zeros((T, N), dtype=np.float32)  # local px within tile

                for t in range(T):
                    h_t = h_all[t].unsqueeze(0).expand(N, -1)  # (N, lstm_h)
                    enc_t = _EncOut(
                        skips=[s.to(device) for s in enc_all.skips],
                        bottom=enc_all.bottom.to(device))
                    pos_maps = model.unet.decode(enc_t, h_t)   # (N, 1, W)
                    pm = pos_maps.squeeze(1)                    # (N, W)
                    tile_scores[t]    = pm.max(dim=-1).values.cpu().numpy()
                    sub_tile_argmax[t] = pm.argmax(dim=-1).float().cpu().numpy()

                # ── DTW → alignment path ──────────────────────────────────
                path_pairs = dtw_backtrack(tile_scores, band_radius_frac=band)
                pred_strip_x = np.zeros(T, dtype=np.float64)
                for t_idx, n_idx in path_pairs:
                    local_x = sub_tile_argmax[t_idx, n_idx]
                    # Scale local_x back to original strip coordinates
                    tile_orig_width = min(W, strip_w - offsets[n_idx])
                    scale = tile_orig_width / W
                    pred_strip_x[t_idx] = offsets[n_idx] + local_x * scale

                # ── Metrics ───────────────────────────────────────────────
                gt_onset   = notes["onset_sec"]
                gt_strip_x = notes["strip_x"]
                frame = np.clip(np.round(gt_onset * eff_hz).astype(int), 0, T - 1)
                pred_at_onset = pred_strip_x[frame]

                m = alignment_metrics(
                    pred_at_onset, gt_strip_x, px_per_sec,
                    beat_times_sec=ann.get("beat_times_sec") or None,
                    bar_times_sec=ann.get("bar_times_sec") or None,
                    gt_onset_sec=gt_onset)
                m.update(henkel_metrics(pred_at_onset, gt_strip_x))
                m["piece_id"] = pid
            except Exception as e:
                m = {"piece_id": pid, "error": repr(e)}

            rows.append(m)
            fout.write(json.dumps(m) + "\n"); fout.flush()
            if (k + 1) % 10 == 0:
                good = [r for r in rows if "error" not in r]
                if good:
                    print(f"  [{k+1}/{len(piece_ids)}] mean_abs_err_sec="
                          f"{np.mean([r['mean_abs_err_sec'] for r in good]):.3f}",
                          flush=True)

    good = [r for r in rows if "error" not in r]
    keys = [kk for kk in good[0]
            if kk.startswith(("mean_", "median_", "pct_", "recall_")) or kk == "n"]
    summ = {"n_pieces": len(good), "n_errors": len(rows) - len(good)}
    for kk in keys:
        vals = np.asarray([r[kk] for r in good if isinstance(r.get(kk), (int, float))])
        if len(vals):
            summ[f"mean_{kk}"] = float(vals.mean())
            summ[f"median_{kk}"] = float(np.median(vals))
    summ["split"] = split; summ["checkpoint"] = checkpoint
    with open(out_root / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--config",     default="configs/v8_henkel_repro.yaml")
    p.add_argument("--processed",  default="data/MSMD/processed")
    p.add_argument("--out_dir",    default=None)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--band",       type=float, default=0.25)
    p.add_argument("--device",     default=None)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device, band=a.band)


if __name__ == "__main__":
    main()
