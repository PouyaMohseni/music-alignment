"""v8 dataset: raw audio.wav + strip.png per piece.

Each __getitem__ returns a random 5-second audio window with the
corresponding strip crop, resized to tile_width for U-Net input.
GT is a Gaussian centred on the ground-truth strip_x position.
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


def crop_and_resize(strip: np.ndarray, cx: int, tile_width: int,
                    strip_width: int) -> np.ndarray:
    """Crop tile_width px centred at cx, resize to tile_width if needed.
    strip: (1, H, W_full)  →  (1, H, tile_width)
    """
    half = tile_width // 2
    x0 = max(0, cx - half)
    x1 = min(strip_width, cx + half)
    crop = strip[:, :, x0:x1]                        # may be narrower at edges
    if crop.shape[-1] != tile_width:
        img = Image.fromarray((crop[0] * 255).astype(np.uint8))
        img = img.resize((tile_width, crop.shape[1]), Image.BILINEAR)
        crop = np.array(img, dtype=np.float32)[np.newaxis] / 255.0
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

    audio_cqt : (1, n_bins, T_win)
    strip_win : (1, H, tile_width)  — grayscale, in [0, 1]
    gt_mask   : (tile_width,)        — Gaussian at strip centre (training target)
    eff_hz    : float                — CQT frame rate
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

        # ── Strip crop centred at gt_x ────────────────────────────────────
        strip = load_strip(piece_dir / "strip.png")        # (1, H, W_full)
        strip_win = crop_and_resize(strip, gt_x, self.tile_width, strip_w)
        # (1, H, tile_width) → squeeze height to (1, tile_width)
        # Average over height axis (strip height is small ~120px)
        strip_win_1d = strip_win.mean(axis=1, keepdims=True)  # (1, 1, tile_width)
        strip_win_1d = strip_win_1d[0]                        # (1, tile_width)

        # ── Gaussian GT — always centred in the tile (we centred on gt_x) ─
        gt_mask = make_gaussian(self.tile_width, self.tile_width // 2, self.sigma_px)

        return {
            "audio_cqt": torch.from_numpy(cqt_win.numpy() if hasattr(cqt_win, 'numpy') else np.array(cqt_win)),
            "strip_win": torch.from_numpy(strip_win_1d),
            "gt_mask":   torch.from_numpy(gt_mask),
            "piece_id":  pid,
            "eff_hz":    float(self.eff_hz),
        }
