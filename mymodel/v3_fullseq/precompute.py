"""Precompute frozen MERT + ViT embeddings + dense per-frame targets per piece.

Because we train over the WHOLE performance (not 5s windows), we cannot
backprop through the foundation models — memory would explode. So we freeze
them and cache their outputs once. Training then only touches the lightweight
projection / cross-attention head.

For each piece in the manifest, writes <emb_root>/<piece_id>.npz with:
    audio_emb   (T, Da)  float16  frozen MERT frame embeddings, pooled to ~eff_hz
    tile_emb    (N, Di)  float16  frozen ViT per-tile embeddings
    pos_tile    (N,)     float32  normalized tile centre position in [0, 1]
    pos_target  (T,)     float32  normalized GT strip position per audio frame
    valid_mask  (T,)     bool     frames within [first_onset, last_onset]
    eff_hz      scalar            effective audio frame rate (native/pool_kernel)
    px_per_sec  scalar            strip_width / duration_sec (for eval)

Usage:
    python -m mymodel.v3_fullseq.precompute \
        --processed data/MSMD/processed \
        --config configs/v3_fullseq.yaml \
        --out data/MSMD/embeddings
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import wave
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from ..v1_baseline.encoders import AudioEncoder, ImageEncoder


class _TarShardWriter:
    """Write per-piece .npz blobs into a few tar shards (avoids inode blowup).

    Produces <out>/shard_000.tar, shard_001.tar, ... and <out>/index.json
    mapping piece_id -> shard filename. ~13x-more-data all-performances runs
    create thousands of pieces; a handful of tar shards keeps the file count low.
    """

    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.index: dict[str, str] = {}
        self.n = 0
        self.shard_idx = 0
        self.tar = None
        self._open()

    def _open(self):
        if self.tar is not None:
            self.tar.close()
        self.cur = f"shard_{self.shard_idx:03d}.tar"
        self.tar = tarfile.open(self.out_dir / self.cur, "w")

    def add(self, piece_id: str, arrays: dict):
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        data = buf.getvalue()
        info = tarfile.TarInfo(name=f"{piece_id}.npz")
        info.size = len(data)
        self.tar.addfile(info, io.BytesIO(data))
        self.index[piece_id] = self.cur
        self.n += 1
        if self.n % self.shard_size == 0:
            self.shard_idx += 1
            self._open()

    def close(self):
        if self.tar is not None:
            self.tar.close()
        with open(self.out_dir / "index.json", "w") as f:
            json.dump(self.index, f)


def _read_wav(path: Path, sr: int) -> np.ndarray:
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == sr, f"expected {sr} Hz, got {r.getframerate()}"
        raw = r.readframes(r.getnframes())
        n_ch, sw = r.getnchannels(), r.getsampwidth()
    dtype = {1: "i1", 2: "<i2", 4: "<i4"}[sw]
    pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 2: pcm /= 32768.0
    elif sw == 4: pcm /= 2147483648.0
    if n_ch > 1: pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    return pcm


@torch.no_grad()
def _encode_audio(audio_enc, audio, sr, chunk_sec, device):
    """Run frozen MERT over the whole performance in chunks; concat pooled frames."""
    chunk = int(chunk_sec * sr)
    outs = []
    for s in range(0, len(audio), chunk):
        seg = audio[s:s + chunk]
        if len(seg) < sr // 2:        # skip <0.5s tail
            break
        a = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(device)
        outs.append(audio_enc(a)[0].cpu())     # (T_chunk, Da)
    return torch.cat(outs, dim=0) if outs else torch.zeros(0, audio_enc.d_audio)


@torch.no_grad()
def _encode_tiles(image_enc, strip, device, sub_batch=128):
    """Run frozen ViT over strip tiles in sub-batches to bound memory."""
    img = torch.from_numpy(np.ascontiguousarray(strip)).permute(2, 0, 1).unsqueeze(0)
    img = img.float() / 255.0
    ts, st = image_enc.tile_size, image_enc.stride
    tiles = img.unfold(3, ts, st)                   # (1,3,H,N,ts)
    N = tiles.size(3)
    tiles = tiles.permute(0, 3, 1, 2, 4).reshape(N, 3, ts, ts)
    feats = []
    for s in range(0, N, sub_batch):
        b = tiles[s:s + sub_batch].to(device)
        out = image_enc.backbone(pixel_values=b).last_hidden_state[:, 0]
        feats.append(out.cpu())
    return torch.cat(feats, dim=0), N               # (N, Di)


def _build_targets(notes, T, eff_hz, strip_w, tile_centres_norm):
    """Dense per-frame normalized target position + validity mask."""
    onset = notes["onset_sec"].astype(np.float64)
    strip_x = notes["strip_x"].astype(np.float64)
    order = np.argsort(onset)
    onset, strip_x = onset[order], strip_x[order]
    times = np.arange(T) / eff_hz
    # monotone interp of strip_x over onset times, clamped to [first, last]
    tgt_px = np.interp(times, onset, strip_x, left=strip_x[0], right=strip_x[-1])
    pos_target = (tgt_px / strip_w).astype(np.float32)
    valid = (times >= onset[0]) & (times <= onset[-1])
    return pos_target, valid.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/MSMD/processed")
    ap.add_argument("--config", default="configs/v3_fullseq.yaml")
    ap.add_argument("--out", default="data/MSMD/embeddings")
    ap.add_argument("--chunk_sec", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--init_checkpoint", default=None,
                    help="v1/v2 checkpoint to load LoRA-adapted encoder weights from")
    ap.add_argument("--lora_rank", type=int, default=0,
                    help="LoRA rank of the init checkpoint's encoders (0 = raw frozen)")
    ap.add_argument("--shard_size", type=int, default=0,
                    help=">0 writes tar shards of this many pieces (avoids inode blowup); "
                         "0 writes individual .npz files")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = cfg.audio_sr
    tile_size, tile_stride = cfg.tile_size, cfg.tile_stride

    audio_enc = AudioEncoder(model_id=cfg.audio_model_id, pool_hz=cfg.pool_hz,
                             freeze=True, lora_rank=args.lora_rank).to(device).eval()
    image_enc = ImageEncoder(model_id=cfg.image_model_id, tile_size=tile_size,
                             stride=tile_stride, freeze=True, lora_rank=args.lora_rank).to(device).eval()
    eff_hz = audio_enc.native_frame_rate / audio_enc.pool_kernel

    # Load LoRA-adapted encoder weights from a trained v1/v2 checkpoint, so the
    # cached embeddings reflect the domain adaptation we trained (not raw frozen).
    if args.init_checkpoint:
        sd = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        params = sd.get("trainable_state", sd.get("model_state", {}))
        a_sd = {k[len("audio_enc."):]: v for k, v in params.items() if k.startswith("audio_enc.")}
        i_sd = {k[len("image_enc."):]: v for k, v in params.items() if k.startswith("image_enc.")}
        ma, ua = audio_enc.load_state_dict(a_sd, strict=False)
        mi, ui = image_enc.load_state_dict(i_sd, strict=False)
        print(f"loaded LoRA from {args.init_checkpoint}  "
              f"audio: {len(a_sd)} keys (missing={len(ma)} unexpected={len(ua)})  "
              f"image: {len(i_sd)} keys (missing={len(mi)} unexpected={len(ui)})", flush=True)

    manifest = [json.loads(l) for l in open(Path(args.processed) / "manifest.jsonl")]
    if args.limit:
        manifest = manifest[: args.limit]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    writer = _TarShardWriter(out_root, args.shard_size) if args.shard_size > 0 else None

    # Tile embeddings are SHARED across all performances of a piece (same strip).
    # Cache the most recent strip so all-performances runs don't recompute ViT
    # N times per piece. Manifest is sorted, so a piece's performances are adjacent.
    tile_cache = {"key": None, "tile_emb": None, "N": None}

    import time
    t0 = time.time()
    done = skipped = 0
    for i, row in enumerate(manifest):
        pid = row["piece_id"]
        if writer is None:
            out_path = out_root / f"{pid}.npz"
            if out_path.exists():
                skipped += 1
                continue
        pdir = Path(args.processed) / pid
        try:
            ann = json.load(open(pdir / "annotations.json"))
            notes = np.load(pdir / "noteheads.npz")
            audio = _read_wav(pdir / "audio.wav", sr)

            audio_emb = _encode_audio(audio_enc, audio, sr, args.chunk_sec, device)
            T = audio_emb.shape[0]

            strip_path = pdir / "strip.png"
            strip_key = os.path.realpath(strip_path)   # shared strips symlink to one file
            if tile_cache["key"] == strip_key:
                tile_emb, N = tile_cache["tile_emb"], tile_cache["N"]
            else:
                strip = np.asarray(Image.open(strip_path).convert("RGB"))
                tile_emb, N = _encode_tiles(image_enc, strip, device)
                tile_cache.update(key=strip_key, tile_emb=tile_emb, N=N)

            strip_w = ann["image"]["width_px"]
            tile_centres = np.arange(N) * tile_stride + tile_size / 2.0
            pos_tile = (tile_centres / strip_w).astype(np.float32)
            pos_target, valid = _build_targets(notes, T, eff_hz, strip_w, pos_tile)
            px_per_sec = strip_w / float(ann["audio"]["duration_sec"])

            arrays = dict(
                audio_emb=audio_emb.numpy().astype(np.float16),
                tile_emb=tile_emb.numpy().astype(np.float16),
                pos_tile=pos_tile,
                pos_target=pos_target,
                valid_mask=valid,
                eff_hz=np.float32(eff_hz),
                px_per_sec=np.float32(px_per_sec),
            )
            if writer is not None:
                writer.add(pid, arrays)
            else:
                np.savez_compressed(out_root / f"{pid}.npz", **arrays)
            done += 1
        except Exception as e:
            print(f"  FAIL {pid}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(manifest)}] done={done} skipped={skipped} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    if writer is not None:
        writer.close()
        print(f"wrote {writer.shard_idx + 1} tar shards + index.json")
    print(f"DONE done={done} skipped={skipped} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
