"""v8 dataset: raw audio.wav + strip.png per piece.

Each __getitem__ returns a random 5-second audio window ending at time t_end,
paired with a strip crop where the GT position (strip_x at t_end) appears
at a RANDOM local offset within the window (not always at center).
This forces the model to learn audio-conditioned position discrimination.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ── CQT helper ────────────────────────────────────────────────────────────────

_CQT_CACHE: dict = {}   # piece_id → CQT tensor, shared across workers via fork

def load_cqt(wav_path: Path, sr: int = 24000,
             n_bins: int = 78, hop: int = 512) -> torch.Tensor:
    """Load audio.wav and compute log-magnitude CQT.
    Returns (1, n_bins, T) float32 tensor.
    Raises FileNotFoundError if audio.wav is missing — synthesise first:
      python -m msmd_prep.run_all --stage synth --processed <processed_root>
    """
    import librosa
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(
            f"audio.wav not found at {wav_path}\n"
            "Run audio synthesis first: python -m msmd_prep.run_all --stage synth")
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    C = librosa.cqt(y, sr=sr, hop_length=hop, n_bins=n_bins, bins_per_octave=12,
                    fmin=librosa.note_to_hz('C1'))
    log_C = np.log1p(np.abs(C)).astype(np.float32)   # (n_bins, T)
    return torch.from_numpy(log_C).unsqueeze(0)       # (1, n_bins, T)


# ── Strip helper ──────────────────────────────────────────────────────────────

def load_strip(strip_path: Path) -> np.ndarray:
    """Load grayscale strip, return float32 (1, H, W) in [0, 1]."""
    img = Image.open(strip_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0    # (H, W)
    return arr[np.newaxis]                            # (1, H, W)


def crop_at_offset(strip: np.ndarray, x0: int, width: int) -> np.ndarray:
    """Crop `width` pixels starting at x0, zero-pad at right edge if needed.
    strip: (1, H, W_full)  →  (1, H, width)
    """
    x1 = x0 + width
    crop = strip[:, :, x0:min(x1, strip.shape[-1])]
    if crop.shape[-1] < width:
        pad = np.zeros((1, strip.shape[1], width - crop.shape[-1]), dtype=np.float32)
        crop = np.concatenate([crop, pad], axis=-1)
    return crop


def make_gaussian(width: int, center: int, sigma: float = 25.0) -> np.ndarray:
    """1-D Gaussian target centred at `center` pixel in [0, width)."""
    x = np.arange(width, dtype=np.float32)
    g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return (g / g.max()).astype(np.float32)           # normalised to [0,1]


# ── Dataset ───────────────────────────────────────────────────────────────────

class HenkelDataset(Dataset):
    """
    One sample = random 5-second audio window + corresponding strip crop.

    audio_cqt  : (1, n_bins, T_win)
    strip_win  : (1, tile_width)     — grayscale 1-D strip window, in [0, 1]
    gt_mask    : (tile_width,)        — Gaussian at actual local GT position
    local_gt_x : int                  — GT pixel offset within the strip window
    eff_hz     : float                — CQT frame rate
    """

    def __init__(self, processed_root: str, split: str,
                 window_sec: float = 5.0, tile_width: int = 512,
                 n_bins: int = 78, hop: int = 512, sr: int = 24000,
                 sigma_px: float = 25.0):
        self.root = Path(processed_root)
        self.window_sec = window_sec
        self.tile_width = tile_width
        self.n_bins = n_bins
        self.hop = hop
        self.sr = sr
        self.sigma_px = sigma_px
        self.eff_hz = sr / hop            # CQT frame rate ≈ 46.875 Hz

        splits = json.load(open(self.root / "splits.json"))
        self.piece_ids = splits[split]

    def __len__(self):
        return len(self.piece_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.piece_ids[idx]
        piece_dir = self.root / pid

        ann = json.load(open(piece_dir / "annotations.json"))
        notes = np.load(piece_dir / "noteheads.npz")
        strip_w = ann["image"]["width_px"]
        dur = float(ann["audio"]["duration_sec"])

        # ── CQT (cached per piece ID) ─────────────────────────────────────
        cqt = _CQT_CACHE.get(pid)
        if cqt is None:
            cqt = load_cqt(piece_dir / "audio.wav",
                           sr=self.sr, n_bins=self.n_bins, hop=self.hop)
            _CQT_CACHE[pid] = cqt

        T_total = cqt.shape[-1]

        # ── Random window: pick end time, back-fill window_sec ───────────
        rng = np.random.default_rng()
        win_frames = int(self.window_sec * self.eff_hz)
        t_end_max = max(win_frames, T_total)
        t_end = rng.integers(win_frames, t_end_max + 1)
        t_start = max(0, t_end - win_frames)
        cqt_win = cqt[:, :, t_start:t_end]   # (1, n_bins, T_win)

        # ── GT strip_x at t_end (find nearest notehead) ──────────────────
        t_end_sec = t_end / self.eff_hz
        onset = notes["onset_sec"]
        if len(onset) == 0:
            gt_x = strip_w // 2
        else:
            nearest = int(np.argmin(np.abs(onset - t_end_sec)))
            gt_x = int(notes["strip_x"][nearest])
        gt_x = np.clip(gt_x, 0, strip_w - 1)

        # ── Strip crop with RANDOM local GT offset ────────────────────────
        # The GT appears at a random position within the window (not always center).
        # The model must use audio context h_t to predict WHERE in the window it is.
        strip = load_strip(piece_dir / "strip.png")        # (1, H, W_full)
        margin = max(16, self.tile_width // 8)
        local_gt_x = int(rng.integers(margin, self.tile_width - margin))

        # Window starting pixel (clamped to valid range)
        window_start = int(np.clip(gt_x - local_gt_x, 0, max(0, strip_w - self.tile_width)))
        actual_local_gt = int(np.clip(gt_x - window_start, 0, self.tile_width - 1))

        strip_win = crop_at_offset(strip, window_start, self.tile_width)  # (1, H, W)
        strip_win_1d = strip_win.mean(axis=1)                              # (1, W)

        gt_mask = make_gaussian(self.tile_width, actual_local_gt, self.sigma_px)

        cqt_arr = cqt_win.numpy() if hasattr(cqt_win, 'numpy') else np.array(cqt_win)
        return {
            "audio_cqt":  torch.from_numpy(cqt_arr),
            "strip_win":  torch.from_numpy(strip_win_1d),
            "gt_mask":    torch.from_numpy(gt_mask),
            "local_gt_x": actual_local_gt,
            "piece_id":   pid,
            "eff_hz":     float(self.eff_hz),
        }
