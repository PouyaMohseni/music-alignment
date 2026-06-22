"""v9 dataset — 2D strip crops, always centered on GT (faithful Henkel training).

Unlike v8 (which used 1D projections), we keep the full 2D strip crop so the
U-Net can see pitch-specific notehead patterns and staff layout. FiLM can then
discriminate based on which notes the audio contains.

Training: crop always centered at GT strip_x → 2D Gaussian at (H/2, W/2).
          Model learns audio↔score visual matching, not "output center."
Inference: causal tracking with crop centered at current x estimate.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_CQT_CACHE: dict = {}


def load_cqt(wav_path: Path, sr: int = 24000,
             n_bins: int = 78, hop: int = 512) -> torch.Tensor:
    """Load audio.wav → log-CQT (1, n_bins, T) float32."""
    import librosa
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(
            f"audio.wav not found at {wav_path}\n"
            "Run: python -m msmd_prep.run_all --stage synth --processed <processed_root>")
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    C = librosa.cqt(y, sr=sr, hop_length=hop, n_bins=n_bins, bins_per_octave=12,
                    fmin=librosa.note_to_hz('C1'))
    log_C = np.log1p(np.abs(C)).astype(np.float32)
    return torch.from_numpy(log_C).unsqueeze(0)   # (1, n_bins, T)


def load_strip_2d(strip_path: Path, h_strip: int) -> np.ndarray:
    """Load grayscale strip, resize height to h_strip.
    Returns float32 (1, h_strip, W_full) in [0, 1].
    """
    img = Image.open(strip_path).convert("L")
    W, H = img.size
    if H != h_strip:
        img = img.resize((W, h_strip), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # (h_strip, W)
    return arr[np.newaxis]                           # (1, h_strip, W)


def crop_2d(strip: np.ndarray, x0: int, tile_width: int) -> np.ndarray:
    """Crop tile_width columns starting at x0; zero-pad at right edge.
    strip: (1, H, W_full) → (1, H, tile_width)
    """
    x1 = x0 + tile_width
    crop = strip[:, :, x0:min(x1, strip.shape[-1])]
    if crop.shape[-1] < tile_width:
        pad = np.zeros((1, strip.shape[1], tile_width - crop.shape[-1]), dtype=np.float32)
        crop = np.concatenate([crop, pad], axis=-1)
    return crop


def make_gaussian_2d(W: int, H: int, cx: int, cy: int,
                     sigma_x: float = 25.0, sigma_y: float = 10.0) -> np.ndarray:
    """2D Gaussian at (cx, cy), normalised to max=1. Returns (H, W) float32."""
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    gx = np.exp(-0.5 * ((x - cx) / sigma_x) ** 2)
    gy = np.exp(-0.5 * ((y - cy) / sigma_y) ** 2)
    g = np.outer(gy, gx)
    return (g / g.max()).astype(np.float32)   # (H, W)


class CPJKUDataset(Dataset):
    """
    One sample = 5-second audio window + 2D score crop centered at GT strip_x.

    audio_cqt  : (1, n_bins, T_win)
    score_crop : (1, h_strip, tile_width)   — 2D grayscale crop in [0, 1]
    gt_mask    : (h_strip, tile_width)      — 2D Gaussian at center (H/2, W/2)
    """

    def __init__(self, processed_root: str, split: str,
                 window_sec: float = 5.0, tile_width: int = 512,
                 h_strip: int = 128, n_bins: int = 78,
                 hop: int = 512, sr: int = 24000,
                 sigma_x: float = 25.0, sigma_y: float = 10.0):
        self.root       = Path(processed_root)
        self.window_sec = window_sec
        self.tile_width = tile_width
        self.h_strip    = h_strip
        self.n_bins     = n_bins
        self.hop        = hop
        self.sr         = sr
        self.sigma_x    = sigma_x
        self.sigma_y    = sigma_y
        self.eff_hz     = sr / hop

        splits = json.load(open(self.root / "splits.json"))
        self.piece_ids = splits[split]

    def __len__(self):
        return len(self.piece_ids)

    def __getitem__(self, idx: int) -> dict:
        pid       = self.piece_ids[idx]
        piece_dir = self.root / pid

        ann     = json.load(open(piece_dir / "annotations.json"))
        notes   = np.load(piece_dir / "noteheads.npz")
        strip_w = ann["image"]["width_px"]

        # ── CQT (cached) ──────────────────────────────────────────────────────
        cqt = _CQT_CACHE.get(pid)
        if cqt is None:
            cqt = load_cqt(piece_dir / "audio.wav",
                           sr=self.sr, n_bins=self.n_bins, hop=self.hop)
            _CQT_CACHE[pid] = cqt
        T_total   = cqt.shape[-1]

        # ── Random 5-second audio window ──────────────────────────────────────
        rng       = np.random.default_rng()
        win_frames = int(self.window_sec * self.eff_hz)
        t_end     = int(rng.integers(win_frames, max(win_frames, T_total) + 1))
        t_start   = max(0, t_end - win_frames)
        cqt_win   = cqt[:, :, t_start:t_end]   # (1, n_bins, T_win)

        # ── GT strip_x at t_end ───────────────────────────────────────────────
        t_end_sec = t_end / self.eff_hz
        onset     = notes["onset_sec"]
        gt_x      = int(notes["strip_x"][int(np.argmin(np.abs(onset - t_end_sec)))]
                        if len(onset) else strip_w // 2)
        gt_x      = int(np.clip(gt_x, 0, strip_w - 1))

        # ── 2D strip crop CENTERED at GT (faithful Henkel training) ───────────
        strip = load_strip_2d(piece_dir / "strip.png", self.h_strip)  # (1, H, W)
        half  = self.tile_width // 2
        x0    = int(np.clip(gt_x - half, 0, max(0, strip_w - self.tile_width)))
        crop  = crop_2d(strip, x0, self.tile_width)   # (1, H, W)

        # GT: 2D Gaussian at center of crop (model matches audio to this visual)
        gt = make_gaussian_2d(self.tile_width, self.h_strip,
                              cx=self.tile_width // 2, cy=self.h_strip // 2,
                              sigma_x=self.sigma_x, sigma_y=self.sigma_y)

        cqt_arr = cqt_win.numpy() if hasattr(cqt_win, "numpy") else np.array(cqt_win)
        return {
            "audio_cqt":  torch.from_numpy(cqt_arr),
            "score_crop": torch.from_numpy(crop),
            "gt_mask":    torch.from_numpy(gt),
            "piece_id":   pid,
        }
