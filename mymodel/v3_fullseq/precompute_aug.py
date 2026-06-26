"""Precompute multi-tempo MERT embeddings for v3_aug retraining.

For each piece in processed/, synthesizes audio at 11 tempo variants using
FluidSynth (same soundfont as CPJKU paper: grand-piano-YDP-20160804.sf2),
runs frozen MERT to get audio_emb, and reuses tile_emb from the existing
single-tempo embedding cache (ViT output doesn't change across tempos).

Output: data/MSMD/embeddings_aug/<piece_id>_tempo_<T_ms>.npz
        data/MSMD/embeddings_aug/splits.json  (train/val/test with all tempo variants)

The tempo variants (ms/beat) match the MSMD-aug dataset:
  500 750 900 950 1000 1050 1100 1250 1500 1750 2000

Usage:
  python -m mymodel.v3_fullseq.precompute_aug \\
      --processed data/MSMD/processed \\
      --emb_cache  data/MSMD/embeddings \\
      --out        data/MSMD/embeddings_aug \\
      --sf         third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import mido
import numpy as np
import torch
from omegaconf import OmegaConf

from ..v1_baseline.encoders import AudioEncoder

TEMPO_VARIANTS_MS = [500, 750, 900, 950, 1000, 1050, 1100, 1250, 1500, 1750, 2000]


def _get_native_tempo_us(midi_path: str) -> int:
    """Return the first tempo event (μs/beat) from a MIDI file, default 500000."""
    mid = mido.MidiFile(midi_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return int(msg.tempo)
    return 500000  # 120 BPM default


def _make_tempo_midi(midi_path: str, target_tempo_us: int, out_path: str):
    """Write a copy of midi_path with the first tempo event replaced by target_tempo_us.

    All subsequent tempo events are removed so the entire piece plays at exactly
    one tempo. This matches the MSMD-aug generation approach.
    """
    mid = mido.MidiFile(midi_path)
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)
        inserted = False
        for msg in track:
            if msg.type == 'set_tempo':
                if not inserted:
                    new_track.append(
                        mido.MetaMessage('set_tempo', tempo=target_tempo_us, time=msg.time))
                    inserted = True
                # drop further tempo events
            else:
                new_track.append(msg.copy())
        if not inserted:
            # No tempo event in track — insert at start of first track
            if new_mid.tracks and new_mid.tracks[0] is new_track:
                new_track.insert(0, mido.MetaMessage('set_tempo', tempo=target_tempo_us, time=0))
    new_mid.save(out_path)


def _synthesize(midi_path: str, sf_path: str, wav_path: str, sr: int = 24000):
    """Synthesize MIDI → WAV using FluidSynth at `sr` Hz."""
    cmd = [
        "fluidsynth", "-F", wav_path, "-O", "s16", "-T", "wav",
        "-r", str(sr), "-q", sf_path, midi_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"fluidsynth failed: {result.stderr.decode()[:200]}")
    if not os.path.exists(wav_path):
        raise RuntimeError(f"fluidsynth produced no output at {wav_path}")


def _read_wav(path: str, sr: int) -> np.ndarray:
    with wave.open(path, "rb") as r:
        actual_sr = r.getframerate()
        raw = r.readframes(r.getnframes())
        n_ch, sw = r.getnchannels(), r.getsampwidth()
    dtype = {1: "i1", 2: "<i2", 4: "<i4"}[sw]
    pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 2:
        pcm /= 32768.0
    elif sw == 4:
        pcm /= 2147483648.0
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    if actual_sr != sr:
        # resample with scipy if needed
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, actual_sr)
        pcm = resample_poly(pcm, sr // g, actual_sr // g).astype(np.float32)
    return pcm


def _wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as r:
        return r.getnframes() / r.getframerate()


@torch.no_grad()
def _encode_audio(audio_enc, audio: np.ndarray, sr: int, chunk_sec: float, device: str):
    chunk = int(chunk_sec * sr)
    outs = []
    for s in range(0, len(audio), chunk):
        seg = audio[s:s + chunk]
        if len(seg) < sr // 2:
            break
        a = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(device)
        outs.append(audio_enc(a)[0].cpu())
    return torch.cat(outs, dim=0) if outs else torch.zeros(0, audio_enc.d_audio)


def _build_targets(onset_sec: np.ndarray, strip_x: np.ndarray,
                   T: int, eff_hz: float, strip_w: int):
    order = np.argsort(onset_sec)
    onset_s = onset_sec[order].astype(np.float64)
    sx = strip_x[order].astype(np.float64)
    times = np.arange(T) / eff_hz
    tgt_px = np.interp(times, onset_s, sx, left=sx[0], right=sx[-1])
    pos_target = (tgt_px / strip_w).astype(np.float32)
    valid = (times >= onset_s[0]) & (times <= onset_s[-1])
    return pos_target, valid.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed",  default="data/MSMD/processed")
    ap.add_argument("--emb_cache",  default="data/MSMD/embeddings",
                    help="Existing single-tempo embeddings (for tile_emb reuse)")
    ap.add_argument("--out",        default="data/MSMD/embeddings_aug")
    ap.add_argument("--sf",         default="third_party/cpjku_unet/audio_conditioned_unet/"
                                            "sound_fonts/grand-piano-YDP-20160804.sf2")
    ap.add_argument("--config",     default="configs/v3_fullseq.yaml")
    ap.add_argument("--chunk_sec",  type=float, default=5.0)
    ap.add_argument("--tempos",     default=None,
                    help="Comma-separated ms/beat values (default: all 11)")
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--workers",    type=int, default=1,
                    help="Parallel FluidSynth workers (synthesis is CPU-bound)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = int(cfg.audio_sr)

    tempo_list = (
        [int(t) for t in args.tempos.split(",")]
        if args.tempos else TEMPO_VARIANTS_MS
    )

    sf_path = str(Path(args.sf).resolve())
    if not os.path.exists(sf_path):
        raise FileNotFoundError(f"SoundFont not found: {sf_path}")

    audio_enc = AudioEncoder(
        model_id=cfg.audio_model_id, pool_hz=cfg.pool_hz,
        freeze=True, lora_rank=0).to(device).eval()
    eff_hz = audio_enc.native_frame_rate / audio_enc.pool_kernel
    print(f"device={device}  sr={sr}  eff_hz={eff_hz:.2f}  tempos={tempo_list}")

    processed = Path(args.processed)
    emb_cache = Path(args.emb_cache)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = [json.loads(l) for l in open(processed / "manifest.jsonl")]
    splits_orig = json.load(open(processed / "splits.json"))

    # Build reverse split lookup: piece_id -> "train"|"val"|"test"
    piece_split = {}
    for split_name, pieces in splits_orig.items():
        for p in pieces:
            piece_split[p] = split_name

    if args.limit:
        manifest = manifest[:args.limit]

    splits_aug: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    t0 = time.time()
    done = skipped = failed = 0

    for i, row in enumerate(manifest):
        pid = row["piece_id"]
        pdir = processed / pid
        split_name = piece_split.get(pid, "train")

        try:
            notes = np.load(pdir / "noteheads.npz")
            onset_sec_native = notes["onset_sec"].astype(np.float64)
            strip_x = notes["strip_x"].astype(np.float64)
            ann = json.load(open(pdir / "annotations.json"))
            strip_w = int(ann["image"]["width_px"])
            midi_path = str(pdir / "score.midi")
            native_tempo_us = _get_native_tempo_us(midi_path)

            # Load tile_emb from single-tempo cache (ViT doesn't change with tempo)
            cache_path = emb_cache / f"{pid}.npz"
            if cache_path.exists():
                cache = np.load(cache_path)
                tile_emb = cache["tile_emb"]    # (N, Di) float16
                pos_tile = cache["pos_tile"]    # (N,) float32
            else:
                print(f"  WARN: no cache for {pid}, skipping tile_emb", flush=True)
                failed += 1
                continue

        except Exception as e:
            print(f"  FAIL {pid} (load): {e}", flush=True)
            failed += 1
            continue

        with tempfile.TemporaryDirectory() as tmp:
            for T_ms in tempo_list:
                out_id = f"{pid}_tempo_{T_ms}"
                out_path = out_root / f"{out_id}.npz"
                if out_path.exists():
                    splits_aug[split_name].append(out_id)
                    skipped += 1
                    continue

                target_tempo_us = T_ms * 1000
                ratio = target_tempo_us / native_tempo_us
                onset_sec_new = onset_sec_native * ratio

                try:
                    mid_path = os.path.join(tmp, f"{T_ms}.mid")
                    wav_path = os.path.join(tmp, f"{T_ms}.wav")
                    _make_tempo_midi(midi_path, target_tempo_us, mid_path)
                    _synthesize(mid_path, sf_path, wav_path, sr=sr)
                    duration = _wav_duration(wav_path)
                    audio = _read_wav(wav_path, sr)
                    audio_emb = _encode_audio(audio_enc, audio, sr, args.chunk_sec, device)
                    T = audio_emb.shape[0]

                    pos_target, valid = _build_targets(
                        onset_sec_new, strip_x, T, eff_hz, strip_w)
                    px_per_sec = strip_w / duration

                    np.savez_compressed(
                        out_path,
                        audio_emb=audio_emb.numpy().astype(np.float16),
                        tile_emb=tile_emb,
                        pos_tile=pos_tile,
                        pos_target=pos_target,
                        valid_mask=valid,
                        eff_hz=np.float32(eff_hz),
                        px_per_sec=np.float32(px_per_sec),
                    )
                    splits_aug[split_name].append(out_id)
                    done += 1
                except Exception as e:
                    print(f"  FAIL {pid} T={T_ms}: {e}", flush=True)
                    failed += 1

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(manifest)}] done={done} skipped={skipped} "
                  f"failed={failed} elapsed={time.time()-t0:.0f}s", flush=True)

    # Write splits.json into out_root (auto-detected by FullSeqDataset)
    with open(out_root / "splits.json", "w") as f:
        json.dump(splits_aug, f, indent=2)

    n = {k: len(v) for k, v in splits_aug.items()}
    print(f"\nDone. done={done} skipped={skipped} failed={failed}  "
          f"elapsed={time.time()-t0:.0f}s")
    print(f"splits: train={n['train']} val={n['val']} test={n['test']}")
    print(f"Output: {out_root}/")


if __name__ == "__main__":
    main()
