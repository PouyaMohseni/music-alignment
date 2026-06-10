"""v3 evaluation: full-sequence alignment, real tracking error in seconds.

    python -m mymodel.v3_fullseq.eval \
        --checkpoint results/v3_fullseq/checkpoint_010000.pt \
        --split test

For each frame the predicted strip position is the expected tile position under
the model's softmax (continuous, sub-tile resolution). Each ground-truth
notehead onset maps to a frame; we compare predicted vs true strip_x and
convert pixel error to seconds via px_per_sec.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .model import FullSeqAlignmentModel, FullSeqModelConfig
from .data import FullSeqDataset, FullSeqTarDataset
from ..shared.metrics import (alignment_metrics, retrieval_metrics, dtw_backtrack,
                              henkel_metrics, dorfer_retrieval_metrics)


def _make_dataset(emb_root, processed_root, split):
    if (Path(emb_root) / "index.json").exists():
        return FullSeqTarDataset(emb_root, processed_root, split)
    return FullSeqDataset(emb_root, processed_root, split)


def _build(cfg, device):
    mc = FullSeqModelConfig(
        d_audio=cfg.model.d_audio, d_image=cfg.model.d_image,
        shared_dim=cfg.model.shared_dim, n_heads=cfg.model.n_heads,
        n_cross_layers=cfg.model.n_cross_layers, dropout=cfg.model.dropout)
    return FullSeqAlignmentModel(mc).to(device)


@torch.no_grad()
def eval_split(checkpoint, cfg_path, processed_root, emb_root, split,
               out_dir=None, limit=None, device=None,
               readout="dtw", band_radius_frac=0.25):
    cfg = OmegaConf.load(cfg_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    temp = cfg.loss.temperature

    model = _build(cfg, device)
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    params = sd.get("trainable_state", sd.get("model_state", {}))
    miss, unexp = model.load_state_dict(params, strict=False)
    if miss or unexp:
        print(f"  load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()

    ds = _make_dataset(emb_root, processed_root, split)
    ids = list(range(len(ds)))[:limit] if limit else list(range(len(ds)))
    out_dir = out_dir or str(Path(checkpoint).parent / "eval")
    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_root / "per_piece.jsonl", "w") as f:
        for k in ids:
            s = ds[k]
            pid = s["piece_id"]
            notes = np.load(Path(processed_root) / pid / "noteheads.npz")
            a = s["audio_emb"].unsqueeze(0).to(device)
            i = s["tile_emb"].unsqueeze(0).to(device)
            sim = model(a, i)["sim"][0]                       # (T, N)

            pos_tile = s["pos_tile"].numpy().astype(np.float64)          # (N,) norm
            eff_hz = s["eff_hz"]
            ann = json.load(open(Path(processed_root) / pid / "annotations.json"))
            strip_w_px = ann["image"]["width_px"]
            px_per_sec = strip_w_px / float(ann["audio"]["duration_sec"])

            sim_np = sim.cpu().numpy()
            if readout == "dtw":
                # Monotonic DTW path over the full similarity matrix — rejects
                # spurious far-away high-similarity peaks (repeated measures).
                path = dtw_backtrack(sim_np, band_radius_frac=band_radius_frac)  # (P,2) of (t,n)
                # one predicted tile per audio frame (last path node at that frame)
                T = sim_np.shape[0]
                pred_tile_per_frame = np.zeros(T, dtype=np.int64)
                for t, n in path:
                    pred_tile_per_frame[t] = n
                pred_px_per_frame = pos_tile[pred_tile_per_frame] * strip_w_px
            else:  # "mean" — expected position (kept for ablation)
                p = F.softmax(sim / temp, dim=-1)
                pred_pos = (p * torch.from_numpy(pos_tile).float().to(device).view(1, -1)).sum(-1).cpu().numpy()
                pred_px_per_frame = pred_pos * strip_w_px

            gt_onset = notes["onset_sec"]; gt_strip_x = notes["strip_x"]
            frame = np.clip(np.round(gt_onset * eff_hz).astype(int), 0, len(pred_px_per_frame) - 1)
            pred_at_onset = pred_px_per_frame[frame]

            m = alignment_metrics(
                pred_at_onset, gt_strip_x, px_per_sec,
                beat_times_sec=ann.get("beat_times_sec") or None,
                bar_times_sec=ann.get("bar_times_sec") or None,
                gt_onset_sec=gt_onset)
            # Henkel 2019 metrics (cm error + global tracking ratio)
            m.update(henkel_metrics(pred_at_onset, gt_strip_x))
            # Dorfer 2017/2018 metrics (Recall@K + MAP)
            m.update(dorfer_retrieval_metrics(sim_np))
            m["piece_id"] = pid
            rows.append(m)
            f.write(json.dumps(m) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"  [{k+1}/{len(ids)}] mean_abs_err_sec="
                      f"{np.mean([r['mean_abs_err_sec'] for r in rows]):.3f}", flush=True)

    keys = [kk for kk in rows[0] if kk.startswith(
        ("mean_", "median_", "pct_", "recall_", "std_", "global_", "tracked_", "map")) or kk == "n"]
    summ = {"n_pieces": len(rows)}
    for kk in keys:
        vals = np.asarray([r[kk] for r in rows if isinstance(r.get(kk), (int, float))])
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
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--config", default="configs/v3_fullseq.yaml")
    p.add_argument("--processed", default="data/MSMD/processed")
    p.add_argument("--emb_root", default="data/MSMD/embeddings")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--readout", default="dtw", choices=["dtw", "mean"],
                   help="dtw = monotonic path (default); mean = expected position")
    p.add_argument("--band_radius_frac", type=float, default=0.25)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.emb_root, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device,
               readout=a.readout, band_radius_frac=a.band_radius_frac)


if __name__ == "__main__":
    main()
