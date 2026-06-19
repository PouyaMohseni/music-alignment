"""v4 eval: pitch-fused alignment, real tracking error in seconds, DTW readout.

    python -m mymodel.v4_pitch.eval --checkpoint results/v4_pitch/checkpoint_010000.pt --split test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .model import PitchFusedModel, PitchFusedConfig
from .data import PitchFusedDataset
from ..shared.metrics import (alignment_metrics, dtw_backtrack,
                              henkel_metrics, retrieval_metrics)


def _build(cfg, device):
    mc = PitchFusedConfig(
        d_audio=cfg.model.d_audio, d_image=cfg.model.d_image, shared_dim=cfg.model.shared_dim,
        n_heads=cfg.model.n_heads, n_cross_layers=cfg.model.n_cross_layers, dropout=cfg.model.dropout,
        pitch_fuse_alpha=cfg.model.pitch_fuse_alpha, pitch_hidden=cfg.model.pitch_hidden)
    return PitchFusedModel(mc).to(device)


@torch.no_grad()
def eval_split(checkpoint, cfg_path, processed_root, emb_root, split,
               out_dir=None, limit=None, device=None, band=0.25):
    cfg = OmegaConf.load(cfg_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _build(cfg, device)
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    params = sd.get("trainable_state", sd.get("model_state", {}))
    miss, unexp = model.load_state_dict(params, strict=False)
    if miss or unexp:
        print(f"  load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()

    ds = PitchFusedDataset(emb_root, processed_root, split, tile_size=cfg.data.tile_size)
    ids = ds.base.piece_ids[:limit] if limit else ds.base.piece_ids
    out_root = Path(out_dir or str(Path(checkpoint).parent / "eval")) / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_root / "per_piece.jsonl", "w") as f:
        for k, pid in enumerate(ids):
            try:
                idx = ds.base.piece_ids.index(pid)
                s = ds[idx]
                a = s["audio_emb"].unsqueeze(0).to(device)
                i = s["tile_emb"].unsqueeze(0).to(device)
                sim = model(a, i)["sim"][0].cpu().numpy()
                pos_tile = s["pos_tile"].numpy().astype(np.float64)
                eff_hz = s["eff_hz"]

                ann = json.load(open(Path(processed_root) / pid / "annotations.json"))
                notes = np.load(Path(processed_root) / pid / "noteheads.npz")
                strip_w = ann["image"]["width_px"]
                px_per_sec = strip_w / float(ann["audio"]["duration_sec"])

                path = dtw_backtrack(sim, band_radius_frac=band)
                T = sim.shape[0]
                pred_tile = np.zeros(T, dtype=np.int64)
                for t, n in path:
                    pred_tile[t] = n
                pred_px = pos_tile[pred_tile] * strip_w

                gt_onset = notes["onset_sec"]; gt_strip_x = notes["strip_x"]
                frame = np.clip(np.round(gt_onset * eff_hz).astype(int), 0, T - 1)
                pred_at_onset = pred_px[frame]

                m = alignment_metrics(
                    pred_at_onset, gt_strip_x, px_per_sec,
                    beat_times_sec=ann.get("beat_times_sec") or None,
                    bar_times_sec=ann.get("bar_times_sec") or None,
                    gt_onset_sec=gt_onset)
                m.update(retrieval_metrics(sim))
                m.update(henkel_metrics(pred_at_onset, gt_strip_x))
                m["piece_id"] = pid
            except Exception as e:
                m = {"piece_id": pid, "error": repr(e)}
            rows.append(m)
            f.write(json.dumps(m) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                good = [r for r in rows if "error" not in r]
                if good:
                    print(f"  [{k+1}/{len(ids)}] mean_abs_err_sec="
                          f"{np.mean([r['mean_abs_err_sec'] for r in good]):.3f}", flush=True)

    good = [r for r in rows if "error" not in r]
    keys = [kk for kk in good[0] if kk.startswith(("mean_", "median_", "pct_", "recall_")) or kk == "n"]
    summ = {"n_pieces": len(good), "n_errors": len(rows) - len(good)}
    for kk in keys:
        vals = np.asarray([r[kk] for r in good if isinstance(r.get(kk), (int, float))])
        if len(vals):
            summ[f"mean_{kk}"] = float(vals.mean()); summ[f"median_{kk}"] = float(np.median(vals))
    summ["split"] = split; summ["checkpoint"] = checkpoint
    with open(out_root / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--config", default="configs/v4_pitch.yaml")
    p.add_argument("--processed", default="data/MSMD/processed")
    p.add_argument("--emb_root", default="data/MSMD/embeddings_lora")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--band", type=float, default=0.25)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.emb_root, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device, band=a.band)


if __name__ == "__main__":
    main()
