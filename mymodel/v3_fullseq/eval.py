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
from .data import FullSeqDataset
from ..shared.metrics import alignment_metrics, retrieval_metrics


def _build(cfg, device):
    mc = FullSeqModelConfig(
        d_audio=cfg.model.d_audio, d_image=cfg.model.d_image,
        shared_dim=cfg.model.shared_dim, n_heads=cfg.model.n_heads,
        n_cross_layers=cfg.model.n_cross_layers, dropout=cfg.model.dropout)
    return FullSeqAlignmentModel(mc).to(device)


@torch.no_grad()
def eval_split(checkpoint, cfg_path, processed_root, emb_root, split,
               out_dir=None, limit=None, device=None):
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

    ds = FullSeqDataset(emb_root, processed_root, split)
    ids = ds.piece_ids[:limit] if limit else ds.piece_ids
    out_dir = out_dir or str(Path(checkpoint).parent / "eval")
    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_root / "per_piece.jsonl", "w") as f:
        for k, pid in enumerate(ids):
            z = np.load(Path(emb_root) / f"{pid}.npz")
            notes = np.load(Path(processed_root) / pid / "noteheads.npz")
            a = torch.from_numpy(z["audio_emb"].astype(np.float32)).unsqueeze(0).to(device)
            i = torch.from_numpy(z["tile_emb"].astype(np.float32)).unsqueeze(0).to(device)
            sim = model(a, i)["sim"][0]                       # (T, N)

            pos_tile = torch.from_numpy(z["pos_tile"]).to(device)        # (N,) norm
            p = F.softmax(sim / temp, dim=-1)                            # (T, N)
            pred_pos = (p * pos_tile.view(1, -1)).sum(dim=-1).cpu().numpy()  # (T,) norm
            eff_hz = float(z["eff_hz"])

            ann = json.load(open(Path(processed_root) / pid / "annotations.json"))
            strip_w_px = ann["image"]["width_px"]
            px_per_sec = strip_w_px / float(ann["audio"]["duration_sec"])

            pred_px_per_frame = pred_pos * strip_w_px                    # (T,)
            gt_onset = notes["onset_sec"]; gt_strip_x = notes["strip_x"]
            frame = np.clip(np.round(gt_onset * eff_hz).astype(int), 0, len(pred_px_per_frame) - 1)
            pred_at_onset = pred_px_per_frame[frame]

            m = alignment_metrics(
                pred_at_onset, gt_strip_x, px_per_sec,
                beat_times_sec=ann.get("beat_times_sec") or None,
                bar_times_sec=ann.get("bar_times_sec") or None,
                gt_onset_sec=gt_onset)
            m.update(retrieval_metrics(sim.cpu().numpy()))
            m["piece_id"] = pid
            rows.append(m)
            f.write(json.dumps(m) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"  [{k+1}/{len(ids)}] mean_abs_err_sec="
                      f"{np.mean([r['mean_abs_err_sec'] for r in rows]):.3f}", flush=True)

    keys = [kk for kk in rows[0] if kk.startswith(("mean_", "median_", "pct_", "recall_")) or kk == "n"]
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
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.emb_root, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == "__main__":
    main()
