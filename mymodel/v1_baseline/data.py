"""Variant A dataloader: random 5-second audio windows + wider strip slices.

The dataset is read from the manifest produced by msmd_prep.manifest. Each
piece is expected to live in its own directory with:

    strip.png         224 px tall, variable width
    audio.wav         mono, 24 kHz, 16-bit PCM (produced by msmd_prep.synth)
    annotations.json  metadata (image dims, mapping, beat times, ...)
    noteheads.npz     columnar arrays (onset_sec, strip_x, ...)

One __getitem__ call returns one stochastic training window:

    audio       (T_samples,)              float32 in [-1, 1]
    image       (3, 224, W_slice)         uint8
    image_mask  (W_slice,)                bool  True for real pixels
    anchors_t   (K,)                      int64  audio bin index per notehead
    anchors_n   (K,)                      int64  image tile index per notehead

collate_fn pads variable-width images and variable-K anchors and emits
batched masks.
"""
from __future__ import annotations
import json, math, os, random
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class WindowConfig:
    audio_sec:        float = 5.0
    audio_sr:         int   = 24000
    pool_hz:          int   = 10
    tile_size:        int   = 224
    tile_stride:      int   = 56
    margin_sec_range: tuple = (3.0, 8.0)   # random per-side strip margin in seconds
    min_anchors:      int   = 3
    max_retries:      int   = 10


@dataclass
class _PieceCache:
    piece_id:        str
    image_path:      str           # strip.png
    audio_path:      str           # audio.wav — loaded lazily per window, not into RAM
    image_width:     int
    image_height:    int           # always 224
    audio_sr:        int
    duration_sec:    float
    pixels_per_sec:  float
    onset_sec:       np.ndarray    # (N,) float32, sorted
    strip_x:         np.ndarray    # (N,) int32
    mapping:         list           # strip_to_page_mapping


# ----------------------------------------------------------------- helpers ---


def _read_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV loader that returns float32 in [-1, 1]."""
    import wave
    with wave.open(path, "rb") as r:
        n_ch  = r.getnchannels()
        n_fr  = r.getnframes()
        sr    = r.getframerate()
        sw    = r.getsampwidth()
        raw   = r.readframes(n_fr)
    dtype = {1: "i1", 2: "<i2", 4: "<i4"}[sw]
    pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 2: pcm /= 32768.0
    elif sw == 4: pcm /= 2147483648.0
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    return pcm, sr


def _load_piece(piece_dir: str, manifest_row: dict) -> _PieceCache:
    with open(os.path.join(piece_dir, manifest_row["annotations"].split("/", 1)[1]
                           if "/" in manifest_row["annotations"] else "annotations.json")) as f:
        ann = json.load(f)
    npz = np.load(os.path.join(piece_dir, "noteheads.npz"))

    audio_path = os.path.join(piece_dir, "audio.wav")
    sr       = ann["audio"]["sample_rate_hz"]
    duration = ann["audio"]["duration_sec"]
    width    = ann["image"]["width_px"]
    height   = ann["image"]["height_px"]

    return _PieceCache(
        piece_id        = ann["piece_id"],
        image_path      = os.path.join(piece_dir, "strip.png"),
        audio_path      = audio_path,
        image_width     = width,
        image_height    = height,
        audio_sr        = sr,
        duration_sec    = duration,
        pixels_per_sec  = width / duration,
        onset_sec       = npz["onset_sec"].astype(np.float32),
        strip_x         = npz["strip_x"].astype(np.int32),
        mapping         = ann["strip_to_page_mapping"],
    )


# --------------------------------------------------------------- the dataset ---


class MSMDAlignmentDataset(Dataset):
    """Stochastic per-call sampler. __len__ is a virtual epoch length."""

    def __init__(
        self,
        manifest_path: str,
        *,
        split: str | None = None,
        window: WindowConfig | None = None,
        seed: int | None = None,
    ):
        self.root      = os.path.dirname(os.path.abspath(manifest_path))
        self.window    = window or WindowConfig()
        self._rng      = random.Random(seed)
        self._np_rng   = np.random.default_rng(seed)

        with open(manifest_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if split:
            rows = [r for r in rows if r["split"] == split]
        if not rows:
            raise ValueError(f"no pieces in manifest (split={split!r})")

        self.pieces: list[_PieceCache] = []
        for row in rows:
            # paths in manifest are relative to manifest dir
            piece_dir = os.path.join(self.root, os.path.dirname(row["image"]))
            if not os.path.exists(os.path.join(piece_dir, "audio.wav")):
                continue
            self.pieces.append(_load_piece(piece_dir, row))
        if not self.pieces:
            raise ValueError("no pieces have audio.wav -- run msmd_prep.synth first")

        self._piece_weights = np.array([p.duration_sec for p in self.pieces], dtype=np.float64)
        self._piece_weights /= self._piece_weights.sum()
        total_sec = sum(p.duration_sec for p in self.pieces)
        self._virtual_length = max(1, int(total_sec // self.window.audio_sec))

    def __len__(self) -> int:
        return self._virtual_length

    # -------------- core sampling --------------
    def _sample_window(self, piece: _PieceCache) -> dict | None:
        w = self.window
        dur = piece.duration_sec
        if dur < w.audio_sec + 0.1:
            return None
        t0 = self._rng.uniform(0.0, dur - w.audio_sec)
        i_lo = int(np.searchsorted(piece.onset_sec, t0, side="left"))
        i_hi = int(np.searchsorted(piece.onset_sec, t0 + w.audio_sec, side="left"))
        if i_hi - i_lo < w.min_anchors:
            return None

        onsets   = piece.onset_sec[i_lo:i_hi]
        strip_xs = piece.strip_x[i_lo:i_hi]

        # asymmetric random strip margin (in seconds → px via piece pixels_per_sec)
        m_lo_sec, m_hi_sec = w.margin_sec_range
        m_left  = self._rng.uniform(m_lo_sec, m_hi_sec)
        m_right = self._rng.uniform(m_lo_sec, m_hi_sec)
        px_per_sec = piece.pixels_per_sec
        x_min = int(strip_xs.min() - m_left  * px_per_sec)
        x_max = int(strip_xs.max() + m_right * px_per_sec)
        x_min = max(0, x_min)
        x_max = min(piece.image_width, x_max)
        if x_max - x_min < w.tile_size + w.tile_stride:
            return None

        # ---- audio slice (lazy: read only the needed window from disk) ----
        sr = piece.audio_sr
        start_sample = int(round(t0 * sr))
        n_samples    = int(round(w.audio_sec * sr))
        import wave as _wave
        with _wave.open(piece.audio_path, "rb") as wf:
            n_ch = wf.getnchannels()
            sw   = wf.getsampwidth()
            wf.setpos(start_sample)
            raw  = wf.readframes(n_samples)
        dtype = {1: "i1", 2: "<i2", 4: "<i4"}[sw]
        pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if sw == 2:   pcm /= 32768.0
        elif sw == 4: pcm /= 2147483648.0
        if n_ch > 1:
            pcm = pcm.reshape(-1, n_ch).mean(axis=1)
        if pcm.shape[0] < n_samples:
            pcm = np.concatenate([pcm, np.zeros(n_samples - pcm.shape[0], dtype=np.float32)])
        audio_slice = pcm

        # ---- image slice (decode strip from disk; rely on OS file cache) ----
        with Image.open(piece.image_path) as im:
            strip = np.array(im.crop((x_min, 0, x_max, piece.image_height)),
                             copy=True)  # (H, W_slice, 3)

        # ---- anchors ----
        # audio bin index at pool_hz
        pool_hz   = w.pool_hz
        t_indices = np.floor((onsets - t0) * pool_hz).astype(np.int64)

        # image tile index: each tile centre sits at stride*k + tile_size/2 in the slice
        rel_x = strip_xs - x_min
        n_indices = ((rel_x - w.tile_size / 2) / w.tile_stride).round().astype(np.int64)
        n_tiles_total = (strip.shape[1] - w.tile_size) // w.tile_stride + 1
        n_indices = np.clip(n_indices, 0, n_tiles_total - 1)

        # Audio bin count must match the model's pooling exactly:
        # native MERT rate is 75 Hz, pool_kernel = round(75 / pool_hz),
        # n_audio_bins = (audio_sec * 75) // pool_kernel.
        native_hz = 75
        pool_kernel = max(1, round(native_hz / pool_hz))
        n_native_frames = int(w.audio_sec * native_hz)
        n_audio_bins = n_native_frames // pool_kernel
        t_indices = np.clip(t_indices, 0, n_audio_bins - 1)

        return {
            "audio":       torch.from_numpy(audio_slice),                    # (T_samples,)
            "image":       torch.from_numpy(strip).permute(2, 0, 1),         # (3, 224, W_slice)
            "anchors_t":   torch.from_numpy(t_indices),
            "anchors_n":   torch.from_numpy(n_indices),
            "piece_id":    piece.piece_id,
            "t0_sec":      t0,
            "x_lo_strip":  x_min,
        }

    def __getitem__(self, idx: int) -> dict:
        for _ in range(self.window.max_retries):
            pi = self._np_rng.choice(len(self.pieces), p=self._piece_weights)
            sample = self._sample_window(self.pieces[int(pi)])
            if sample is not None:
                return sample
        raise RuntimeError(
            f"could not find a valid window in {self.window.max_retries} tries; "
            "check min_anchors / margin range"
        )


# ----------------------------------------------------------------- collate ---


def collate_fn(batch: list[dict]) -> dict:
    B = len(batch)
    audio = torch.stack([b["audio"] for b in batch], dim=0)             # (B, T_samples)

    H = batch[0]["image"].shape[1]
    Wmax = max(b["image"].shape[2] for b in batch)
    image = torch.zeros(B, 3, H, Wmax, dtype=torch.uint8)
    image_mask = torch.zeros(B, Wmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        w = b["image"].shape[2]
        image[i, :, :, :w] = b["image"]
        image_mask[i, :w] = True

    Kmax = max(b["anchors_t"].shape[0] for b in batch)
    anchors_t = torch.full((B, Kmax), -1, dtype=torch.long)
    anchors_n = torch.full((B, Kmax), -1, dtype=torch.long)
    anchor_mask = torch.zeros(B, Kmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b["anchors_t"].shape[0]
        anchors_t[i, :k]   = b["anchors_t"]
        anchors_n[i, :k]   = b["anchors_n"]
        anchor_mask[i, :k] = True

    return {
        "audio":        audio,
        "image":        image,
        "image_mask":   image_mask,
        "anchors_t":    anchors_t,
        "anchors_n":    anchors_n,
        "anchor_mask":  anchor_mask,
        "piece_ids":    [b["piece_id"] for b in batch],
        "t0_sec":       torch.tensor([b["t0_sec"] for b in batch]),
        "x_lo_strip":   torch.tensor([b["x_lo_strip"] for b in batch]),
    }


# --------------------------------------------------------------- factories ---


def build_dataloaders(
    manifest_path: str,
    *,
    window: WindowConfig | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    out = {}
    for split in ("train", "val"):
        try:
            ds = MSMDAlignmentDataset(manifest_path, split=split, window=window, seed=seed)
        except ValueError:
            continue
        out[split] = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    return out
