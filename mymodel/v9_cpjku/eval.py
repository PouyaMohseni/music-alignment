"""v9 eval — causal tracking with 2D score crops.

Inference:
  1. audio.wav → CQT → AudioCNN → LSTM → h[0..T-1]
  2. Causal tracking: for each batch of frames,
       crop (1, h_strip, tile_width) centered at current x estimate
       → UNet2D(crop, h_t) → (1, h_strip, tile_width) heatmap
       → sum over H → 1D profile → argmax → new x estimate

    python -m mymodel.v9_cpjku.eval \
        --checkpoint results/v9_cpjku/checkpoint_030000.pt \
        --config configs/v9_cpjku.yaml --split test \
        --processed /project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .model import CPJKU, CPJKUConfig
from .data import load_cqt, load_strip_2d, crop_2d
from ..shared.metrics import alignment_metrics, henkel_metrics


def _build(cfg, device):
    hc = CPJKUConfig(
        n_bins=cfg.model.n_bins,
        cnn_channels=list(cfg.model.cnn_channels),
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        lstm_bidirectional=cfg.model.get("lstm_bidirectional", False),
        unet_channels=list(cfg.model.unet_channels),
        h_strip=cfg.model.h_strip,
        tile_width=cfg.model.tile_width)
    return CPJKU(hc).to(device)


@torch.no_grad()
def eval_split(checkpoint, cfg_path, processed_root, split,
               out_dir=None, limit=None, device=None, batch_frames=32):
    cfg    = OmegaConf.load(cfg_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = _build(cfg, device)
    sd     = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"], strict=True)
    model.eval()

    hop    = cfg.data.hop
    sr     = cfg.data.sr
    W      = cfg.model.tile_width
    H      = cfg.model.h_strip
    eff_hz = sr / hop

    proc   = Path(processed_root)
    piece_ids = json.load(open(proc / "splits.json"))[split]
    if limit:
        piece_ids = piece_ids[:limit]

    out_root = Path(out_dir or str(Path(checkpoint).parent / "eval")) / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_root / "per_piece.jsonl", "w") as fout:
        for k, pid in enumerate(piece_ids):
            try:
                piece_dir = proc / pid
                ann       = json.load(open(piece_dir / "annotations.json"))
                notes     = np.load(piece_dir / "noteheads.npz")
                strip_w   = ann["image"]["width_px"]
                dur       = float(ann["audio"]["duration_sec"])
                px_per_sec = strip_w / dur

                # ── Audio → context vectors ────────────────────────────────
                cqt   = load_cqt(piece_dir / "audio.wav",
                                  sr=sr, n_bins=cfg.model.n_bins, hop=hop)
                T     = cqt.shape[-1]
                h_all = model.audio_context(cqt.unsqueeze(0).to(device))[0]  # (T, lstm_h)

                # ── Load 2D strip ──────────────────────────────────────────
                strip_2d = load_strip_2d(piece_dir / "strip.png", H)  # (1, H, W_full)

                # ── Causal tracking ────────────────────────────────────────
                pos_estimate = 0.0
                pred_strip_x = np.zeros(T, dtype=np.float64)

                for t_start in range(0, T, batch_frames):
                    t_end_b = min(t_start + batch_frames, T)
                    B = t_end_b - t_start

                    # Crop centered at current estimate (same window for this batch)
                    cx = int(round(pos_estimate))
                    x0 = max(0, min(strip_w - W, cx - W // 2))

                    crop    = crop_2d(strip_2d, x0, W)                  # (1, H, W)
                    crop_t  = torch.from_numpy(
                        np.tile(crop[np.newaxis], (B, 1, 1, 1))          # (B, 1, H, W)
                    ).to(device)

                    h_batch  = h_all[t_start:t_end_b]                    # (B, lstm_h)
                    pos_maps = model.unet(crop_t, h_batch)                # (B, 1, H, W)

                    # Collapse H → 1D profile, argmax → local x
                    col      = pos_maps.squeeze(1).sum(dim=1)             # (B, W)
                    local_xs = col.argmax(dim=-1).cpu().numpy()           # (B,)

                    for i, local_x in enumerate(local_xs):
                        new_pos = float(np.clip(x0 + local_x, 0, strip_w - 1))
                        pred_strip_x[t_start + i] = new_pos
                        pos_estimate = new_pos

                # ── Metrics ────────────────────────────────────────────────
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
                import traceback
                m = {"piece_id": pid, "error": repr(e), "tb": traceback.format_exc()}

            rows.append(m)
            fout.write(json.dumps(m) + "\n"); fout.flush()
            if (k + 1) % 10 == 0:
                good = [r for r in rows if "error" not in r]
                if good:
                    print(f"  [{k+1}/{len(piece_ids)}] mean_abs_err_sec="
                          f"{np.mean([r['mean_abs_err_sec'] for r in good]):.3f}",
                          flush=True)

    good = [r for r in rows if "error" not in r]
    if not good:
        for r in [x for x in rows if "error" in x][:3]:
            print(f"  {r['piece_id']}: {r['error']}\n{r.get('tb','')}")
        return None

    keys = [k for k in good[0]
            if k.startswith(("mean_", "median_", "pct_", "recall_")) or k == "n"]
    summ = {"n_pieces": len(good), "n_errors": len(rows) - len(good)}
    for k in keys:
        vals = np.asarray([r[k] for r in good if isinstance(r.get(k), (int, float))])
        if len(vals):
            summ[f"mean_{k}"] = float(vals.mean())
            summ[f"median_{k}"] = float(np.median(vals))
    summ["split"] = split; summ["checkpoint"] = checkpoint
    with open(out_root / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--config",     default="configs/v9_cpjku.yaml")
    p.add_argument("--processed",  default="data/MSMD/processed")
    p.add_argument("--out_dir",    default=None)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--device",     default=None)
    p.add_argument("--batch_frames", type=int, default=32)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device,
               batch_frames=a.batch_frames)


if __name__ == "__main__":
    main()
