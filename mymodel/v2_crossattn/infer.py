"""v2 inference: load checkpoint, run cross-attention alignment, save results.

    python -m mymodel.v2_crossattn.infer \
        --checkpoint results/v2_nce/checkpoint_004000.pt \
        --piece_id BachCPE__cpe-bach-rondo__cpe-bach-rondo

    python -m mymodel.v2_crossattn.eval \
        --checkpoint results/v2_nce/checkpoint_004000.pt \
        --split test
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from .model import CrossAttnAlignmentModel, CrossAttnModelConfig
from ..shared.metrics import alignment_metrics, dtw_backtrack, retrieval_metrics, henkel_metrics, dorfer_retrieval_metrics


def _read_wav(path: Path, sr: int) -> np.ndarray:
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


def _build_model(cfg, device) -> CrossAttnAlignmentModel:
    mc = CrossAttnModelConfig(
        shared_dim=cfg.model.shared_dim,
        n_heads=cfg.model.get("n_heads", 4),
        attn_dropout=cfg.model.get("attn_dropout", 0.1),
        head_dropout=cfg.model.head_dropout,
        audio_model_id=cfg.model.audio_model_id,
        pool_hz=cfg.model.pool_hz,
        freeze_audio=cfg.model.freeze_audio,
        lora_rank_audio=cfg.model.get("lora_rank_audio", 0),
        image_model_id=cfg.model.image_model_id,
        tile_size=cfg.model.tile_size,
        tile_stride=cfg.model.tile_stride,
        freeze_image=cfg.model.freeze_image,
        lora_rank_image=cfg.model.get("lora_rank_image", 0),
    )
    return CrossAttnAlignmentModel(mc).to(device)


@torch.no_grad()
def align_piece(
    piece_id: str,
    checkpoint_path: str,
    *,
    processed_root: str = "data/MSMD/processed",
    config_path: str = "configs/v2_crossattn.yaml",
    out_dir: str = "results/v2_nce/infer",
    chunk_sec: float = 20.0,
    band_radius_frac: float = 0.5,
    device: str | None = None,
) -> dict:
    cfg = OmegaConf.load(config_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    piece_dir = Path(processed_root) / piece_id
    ann = json.load(open(piece_dir / "annotations.json"))
    notes = np.load(piece_dir / "noteheads.npz")

    strip = np.asarray(Image.open(piece_dir / "strip.png").convert("RGB"))
    H, W, _ = strip.shape
    image = torch.from_numpy(np.ascontiguousarray(strip)).permute(2, 0, 1).unsqueeze(0)

    audio_sr = cfg.window.audio_sr
    pool_hz = cfg.window.pool_hz
    wav_path = piece_dir / "audio.wav"
    if wav_path.exists():
        audio = _read_wav(wav_path, audio_sr)
        audio_source = "wav"
    else:
        dur = float(ann["audio"]["duration_sec"])
        audio = np.zeros(int(dur * audio_sr), dtype=np.float32)
        audio_source = "zeros"
        print(f"  WARN {piece_id}: no audio.wav")

    model = _build_model(cfg, device)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = sd.get("trainable_state", sd.get("model_state", {}))
    missing, unexpected = model.load_state_dict(params, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    image = image.to(device)

    # Encode full strip once — cross-attend against each audio chunk
    i_feats, tile_mask = model.image_enc(image)
    i_proj = model.image_proj(i_feats)                     # (1, N, d) before cross-attn

    chunk_samples = int(chunk_sec * audio_sr)
    sim_chunks: list[np.ndarray] = []
    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    for ci in range(n_chunks):
        s = ci * chunk_samples
        e = min(s + chunk_samples, len(audio))
        a_raw = torch.from_numpy(audio[s:e].astype(np.float32)).unsqueeze(0).to(device)
        a_feats = model.audio_enc(a_raw)
        a_proj = model.audio_proj(a_feats)                 # (1, T_chunk, d)

        # Cross-attention: audio ↔ image
        a_ctx = model.audio_cross(query=a_proj, context=i_proj)   # (1, T, d)
        i_ctx = model.image_cross(query=i_proj, context=a_proj)   # (1, N, d)

        import torch.nn.functional as F
        a_norm = F.normalize(a_ctx, dim=-1)
        i_norm = F.normalize(i_ctx, dim=-1)
        sim = torch.einsum("btd,bnd->btn", a_norm, i_norm)[0].cpu().numpy()
        sim_chunks.append(sim)

    sim = np.concatenate(sim_chunks, axis=0)                # (T_total, N)

    path = dtw_backtrack(sim, band_radius_frac=band_radius_frac)
    N = sim.shape[1]
    tile_stride = cfg.model.tile_stride
    tile_size = cfg.model.tile_size
    tile_centres = np.arange(N) * tile_stride + tile_size // 2
    pred_strip_x_per_frame = tile_centres[path[:, 1]]

    gt_onset_sec = notes["onset_sec"]
    gt_strip_x = notes["strip_x"]
    onset_pool_frame = np.round(gt_onset_sec * pool_hz).astype(np.int64)
    onset_pool_frame = np.clip(onset_pool_frame, 0, path[:, 0].max())
    last_idx = np.searchsorted(path[:, 0], onset_pool_frame, side="right") - 1
    last_idx = np.clip(last_idx, 0, len(path) - 1)
    pred_at_onset = pred_strip_x_per_frame[last_idx]

    strip_w = ann["image"]["width_px"]
    px_per_sec = strip_w / float(ann["audio"]["duration_sec"])
    beat_times = ann.get("beat_times_sec") or []
    bar_times = ann.get("bar_times_sec") or []

    metrics = alignment_metrics(
        pred_at_onset, gt_strip_x, px_per_sec,
        beat_times_sec=beat_times if beat_times else None,
        bar_times_sec=bar_times if bar_times else None,
        gt_onset_sec=gt_onset_sec,
    )
    metrics.update(retrieval_metrics(sim))
    metrics.update(henkel_metrics(pred_at_onset, gt_strip_x))
    metrics.update(dorfer_retrieval_metrics(sim))
    metrics["piece_id"] = piece_id
    metrics["pixels_per_sec"] = float(px_per_sec)
    metrics["sim_shape"] = list(sim.shape)
    metrics["audio_source"] = audio_source

    out_root = Path(out_dir) / piece_id
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_root / "path.npz",
        audio_frame=path[:, 0].astype(np.int32),
        tile_idx=path[:, 1].astype(np.int32),
        strip_x_pred=pred_strip_x_per_frame.astype(np.int32),
        sim=sim.astype(np.float32),
    )
    with open(out_root / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    crop_w = min(1500, strip_w)
    overlay = Image.fromarray(strip[:, :crop_w]).copy()
    dr = ImageDraw.Draw(overlay)
    for gx in gt_strip_x[gt_strip_x < crop_w]:
        dr.ellipse([gx - 3, 0, gx + 3, 6], outline="red", width=1)
    for px in pred_at_onset[gt_strip_x < crop_w]:
        dr.ellipse([px - 3, H - 7, px + 3, H - 1], outline="green", width=1)
    overlay.save(out_root / "overlay.png")

    print(json.dumps(metrics, indent=2))
    return metrics


def _parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--piece_id", required=True)
    p.add_argument("--processed", default="data/MSMD/processed")
    p.add_argument("--config", default="configs/v2_crossattn.yaml")
    p.add_argument("--out_dir", default="results/v2_nce/infer")
    p.add_argument("--chunk_sec", type=float, default=20.0)
    p.add_argument("--band_radius_frac", type=float, default=0.5)
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_cli()
    align_piece(a.piece_id, a.checkpoint, processed_root=a.processed,
                config_path=a.config, out_dir=a.out_dir,
                chunk_sec=a.chunk_sec, band_radius_frac=a.band_radius_frac,
                device=a.device)
