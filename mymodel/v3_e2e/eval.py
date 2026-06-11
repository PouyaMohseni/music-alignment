"""v3_e2e evaluation: run live encoders over full audio+strip, DTW alignment.

    python -m mymodel.v3_e2e.eval \
        --checkpoint results/v3_e2e/checkpoint_000500.pt \
        --split test
"""
from __future__ import annotations
import argparse, json, wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from .model import E2EAlignmentModel, E2EModelConfig
from ..shared.metrics import (alignment_metrics, retrieval_metrics,
                               dtw_backtrack, henkel_metrics,
                               dorfer_retrieval_metrics)


def _read_wav(path, sr):
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == sr
        raw = r.readframes(r.getnframes())
        n_ch, sw = r.getnchannels(), r.getsampwidth()
    dtype = {1: "i1", 2: "<i2", 4: "<i4"}[sw]
    pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 2: pcm /= 32768.0
    elif sw == 4: pcm /= 2147483648.0
    if n_ch > 1: pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    return pcm


def _build(cfg, device):
    mc = E2EModelConfig(
        audio_model_id=cfg.model.audio_model_id,
        pool_hz=cfg.window.pool_hz,
        lora_rank_audio=cfg.model.lora_rank_audio,
        image_model_id=cfg.model.image_model_id,
        tile_size=cfg.window.tile_size,
        tile_stride=cfg.window.tile_stride,
        lora_rank_image=0,   # image enc frozen at train time
        shared_dim=cfg.model.shared_dim,
        n_heads=cfg.model.n_heads,
        n_cross_layers=cfg.model.n_cross_layers,
        dropout=cfg.model.dropout,
    )
    return E2EAlignmentModel(mc).to(device)


@torch.no_grad()
def eval_split(checkpoint, cfg_path, processed_root, split,
               out_dir=None, limit=None, device=None,
               chunk_sec=20.0, band_radius_frac=0.25):
    cfg = OmegaConf.load(cfg_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = _build(cfg, device)
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    params = sd.get("trainable_state", sd.get("model_state", {}))
    miss, unexp = model.load_state_dict(params, strict=False)
    if miss or unexp:
        print(f"  load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()

    splits = json.load(open(Path(processed_root) / "splits.json"))
    ids = splits[split][:limit] if limit else splits[split]
    out_dir = out_dir or str(Path(checkpoint).parent / "eval")
    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)

    sr = cfg.window.audio_sr
    pool_hz = cfg.window.pool_hz
    tile_stride = cfg.window.tile_stride
    tile_size = cfg.window.tile_size
    chunk_samples = int(chunk_sec * sr)

    rows = []
    with open(out_root / "per_piece.jsonl", "w") as f:
        for k, pid in enumerate(ids):
            pdir = Path(processed_root) / pid
            ann = json.load(open(pdir / "annotations.json"))
            notes = np.load(pdir / "noteheads.npz")
            strip = np.asarray(Image.open(pdir / "strip.png").convert("RGB"))
            audio = _read_wav(pdir / "audio.wav", sr)

            # encode strip once (frozen ViT, no grad)
            img = torch.from_numpy(np.ascontiguousarray(strip)).permute(2, 0, 1).unsqueeze(0).to(device)
            i_feat, _ = model.image_enc(img)
            i_proj = model.image_proj(i_feat)   # (1, N, d)
            N = i_proj.shape[1]
            tile_centres = np.arange(N) * tile_stride + tile_size // 2

            # chunk audio through MERT + cross-attend with strip
            sim_chunks = []
            for s in range(0, len(audio), chunk_samples):
                seg = audio[s:min(s + chunk_samples, len(audio))]
                if len(seg) < sr // 2:
                    break
                a = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(device)
                a_feat = model.audio_enc(a)
                a_proj = model.audio_proj(a_feat)
                for la, li in zip(model.audio_cross, model.image_cross):
                    a_ctx = la(query=a_proj, context=i_proj)
                    i_ctx = li(query=i_proj, context=a_proj)
                    a_proj, i_proj_local = a_ctx, i_ctx
                a_norm = F.normalize(a_proj, dim=-1)
                i_norm = F.normalize(i_proj_local, dim=-1)
                sim_chunks.append(torch.einsum("btd,bnd->btn",
                                               a_norm, i_norm)[0].cpu().numpy())
            if not sim_chunks:
                continue
            sim = np.concatenate(sim_chunks, axis=0)

            path = dtw_backtrack(sim, band_radius_frac=band_radius_frac)
            pred_strip_x = tile_centres[path[:, 1]]

            gt_onset = notes["onset_sec"]
            gt_strip_x = notes["strip_x"]
            strip_w = ann["image"]["width_px"]
            px_per_sec = strip_w / float(ann["audio"]["duration_sec"])
            frame = np.clip(np.round(gt_onset * pool_hz).astype(int),
                            0, path[:, 0].max())
            last_idx = np.searchsorted(path[:, 0], frame, side="right") - 1
            last_idx = np.clip(last_idx, 0, len(path) - 1)
            pred_at_onset = pred_strip_x[last_idx]

            m = alignment_metrics(pred_at_onset, gt_strip_x, px_per_sec,
                                   beat_times_sec=ann.get("beat_times_sec") or None,
                                   bar_times_sec=ann.get("bar_times_sec") or None,
                                   gt_onset_sec=gt_onset)
            m.update(henkel_metrics(pred_at_onset, gt_strip_x))
            m.update(dorfer_retrieval_metrics(sim))
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
    p.add_argument("--config", default="configs/v3_e2e.yaml")
    p.add_argument("--processed", default="data/MSMD/processed")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--chunk_sec", type=float, default=20.0)
    p.add_argument("--band_radius_frac", type=float, default=0.25)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device,
               chunk_sec=a.chunk_sec, band_radius_frac=a.band_radius_frac)


if __name__ == "__main__":
    main()
