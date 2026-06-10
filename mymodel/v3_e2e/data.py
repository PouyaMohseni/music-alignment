"""v3_e2e data loader: windowed audio + full-strip score + dense targets.

Each sample:
    audio_window  (T_samples,)    5 s mono waveform at 24 kHz
    strip         (3, 224, W)     full-strip image (uint8)
    pos_tile      (N,)            normalized tile centre positions [0,1]
    pos_target    (T_pool,)       normalized GT strip position per pooled frame
    valid_mask    (T_pool,)       bool — frames within [first_onset, last_onset]
    eff_hz        float           pooled audio frame rate

Strip is loaded lazily and cached in memory per piece (it never changes).
"""
from __future__ import annotations
import json
import random
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def _read_wav_slice(path, sr, start_sample: int, n_samples: int) -> np.ndarray:
    """Read only the needed slice from a WAV — avoids loading the full file."""
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == sr
        n_ch, sw, n_frames = r.getnchannels(), r.getsampwidth(), r.getnframes()
        start = max(0, min(start_sample, n_frames))
        count = min(n_samples, n_frames - start)
        r.setpos(start)
        raw = r.readframes(count)
    dtype = {1: "i1", 2: "<i2", 4: "<i4"}[sw]
    pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 2: pcm /= 32768.0
    elif sw == 4: pcm /= 2147483648.0
    if n_ch > 1: pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    # pad if file ended before n_samples
    if len(pcm) < n_samples:
        pcm = np.pad(pcm, (0, n_samples - len(pcm)))
    return pcm


def _read_wav_duration(path, sr) -> float:
    """Read just the header to get duration — no sample data loaded."""
    with wave.open(str(path), "rb") as r:
        return r.getnframes() / r.getframerate()


class E2EDataset(Dataset):
    """One __getitem__ = one random 5-second window from a random piece."""

    def __init__(self, processed_root: str, split: str, *,
                 audio_sec: float = 5.0, audio_sr: int = 24000,
                 pool_hz: int = 10, tile_size: int = 224,
                 tile_stride: int = 56, seed: int | None = None):
        self.root = Path(processed_root)
        self.audio_sec = audio_sec
        self.audio_sr = audio_sr
        self.pool_hz = pool_hz
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        splits = json.load(open(self.root / "splits.json"))
        pids = splits[split]
        self.pieces = []
        for pid in pids:
            pdir = self.root / pid
            wav = pdir / "audio.wav"
            if not wav.exists():
                continue
            # read duration from WAV header only (no sample data)
            dur = _read_wav_duration(wav, audio_sr)
            if dur < audio_sec + 0.5:
                continue
            ann = json.load(open(pdir / "annotations.json"))
            notes = np.load(pdir / "noteheads.npz")
            self.pieces.append({
                "pid": pid,
                "pdir": pdir,
                "dur": dur,
                "strip_w": ann["image"]["width_px"],
                "onset_sec": notes["onset_sec"].astype(np.float32),
                "strip_x": notes["strip_x"].astype(np.float32),
            })

        if not self.pieces:
            raise ValueError(f"no usable pieces for split={split}")

        self._weights = np.array([p["dur"] for p in self.pieces])
        self._weights /= self._weights.sum()
        self._strip_cache: dict[str, np.ndarray] = {}
        # Audio is NOT cached — 7005 pieces × ~5 MB = ~35 GB would OOM.
        # Strips ARE cached — only 467 unique strips (shared across performances) × ~6 MB = ~3 GB.

        # virtual epoch: roughly total_seconds / audio_sec
        total = sum(p["dur"] for p in self.pieces)
        self._len = max(1, int(total // audio_sec))

    def __len__(self):
        return self._len

    def _get_strip(self, p):
        pid = p["pid"]
        if pid not in self._strip_cache:
            img = Image.open(p["pdir"] / "strip.png").convert("RGB")
            self._strip_cache[pid] = np.asarray(img)
        return self._strip_cache[pid]

    def _get_audio_slice(self, p, start_sample: int, n_samples: int) -> np.ndarray:
        return _read_wav_slice(p["pdir"] / "audio.wav", self.audio_sr,
                               start_sample, n_samples)

    def __getitem__(self, idx):
        pi = int(self._np_rng.choice(len(self.pieces), p=self._weights))
        p = self.pieces[pi]

        strip = self._get_strip(p)

        # sample a random window start, read only that slice from disk
        t0 = self._rng.uniform(0.0, p["dur"] - self.audio_sec)
        s = int(round(t0 * self.audio_sr))
        n_samples = int(self.audio_sec * self.audio_sr)
        window = self._get_audio_slice(p, s, n_samples)

        # pooled frame count
        native_hz = 75
        pool_kernel = max(1, round(native_hz / self.pool_hz))
        T_pool = int(self.audio_sec * native_hz) // pool_kernel

        # strip tile positions
        H, W, _ = strip.shape
        N = (W - self.tile_size) // self.tile_stride + 1
        tile_centres = (np.arange(N) * self.tile_stride
                        + self.tile_size / 2.0).astype(np.float32)
        pos_tile = tile_centres / p["strip_w"]

        # dense per-frame GT target via monotone interpolation of onsets
        onset = p["onset_sec"].astype(np.float64)
        sx = p["strip_x"].astype(np.float64)
        order = np.argsort(onset)
        onset, sx = onset[order], sx[order]
        frame_times = t0 + np.arange(T_pool) / self.pool_hz
        tgt_px = np.interp(frame_times, onset, sx,
                           left=sx[0], right=sx[-1]).astype(np.float32)
        pos_target = tgt_px / p["strip_w"]
        valid = ((frame_times >= onset[0]) & (frame_times <= onset[-1]))

        return {
            "audio_window": torch.from_numpy(window),
            "strip":        torch.from_numpy(np.ascontiguousarray(strip)).permute(2, 0, 1),
            "pos_tile":     torch.from_numpy(pos_tile),
            "pos_target":   torch.from_numpy(pos_target),
            "valid_mask":   torch.from_numpy(valid.astype(bool)),
            "eff_hz":       float(self.pool_hz),
            "piece_id":     p["pid"],
        }


def collate(batch):
    # strip widths vary — pad to max width in batch
    B = len(batch)
    Wmax = max(b["strip"].shape[2] for b in batch)
    strips = torch.zeros(B, 3, batch[0]["strip"].shape[1], Wmax, dtype=torch.uint8)
    for i, b in enumerate(batch):
        w = b["strip"].shape[2]
        strips[i, :, :, :w] = b["strip"]

    return {
        "audio_window": torch.stack([b["audio_window"] for b in batch]),
        "strip":        strips,
        "pos_tile":     [b["pos_tile"] for b in batch],      # variable N per piece
        "pos_target":   torch.stack([b["pos_target"] for b in batch]),
        "valid_mask":   torch.stack([b["valid_mask"] for b in batch]),
        "piece_ids":    [b["piece_id"] for b in batch],
    }


def build_loaders(processed_root, split_names, *, audio_sec=5.0, audio_sr=24000,
                  pool_hz=10, tile_size=224, tile_stride=56,
                  batch_size=4, num_workers=4, seed=42):
    out = {}
    for split in split_names:
        try:
            ds = E2EDataset(processed_root, split, audio_sec=audio_sec,
                            audio_sr=audio_sr, pool_hz=pool_hz,
                            tile_size=tile_size, tile_stride=tile_stride,
                            seed=seed)
        except ValueError:
            continue
        out[split] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate,
                                persistent_workers=num_workers > 0)
    return out
