"""v9 data loader — adapts our MSMD processed format to the CPJKU model.

Their spectrogram: sr=22050, fps=20 (hop=1102), 78-bin log-mel, 60-6000 Hz.
We compute this from audio.wav using librosa (no madmom dependency).

Their GT: binary rectangle mask at the current position (not Gaussian).
Their score: 2D crop centered at GT, always (training) or at estimate (eval).

Their CBEncoder takes 40 frames (=2 seconds at fps=20) as context window.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_SPEC_CACHE: dict = {}   # piece_id → (78, T) spectrogram


def load_spec(wav_path: Path, sr_out: int = 22050,
              fps: int = 20, n_mels: int = 78,
              fmin: float = 60.0, fmax: float = 6000.0) -> np.ndarray:
    """Load audio.wav and compute log-mel spectrogram matching CPJKU format.

    Returns (n_mels, T) float32, normalised similarly to their log-spec.
    fps=20, sr=22050 matches their spectrogram_params exactly.
    """
    import librosa
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(
            f"audio.wav not found: {wav_path}\n"
            "Run: python -m msmd_prep.run_all --stage synth")
    y, _ = librosa.load(str(wav_path), sr=sr_out, mono=True)
    hop = int(sr_out / fps)   # 22050/20 = 1102
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr_out, n_fft=2048, hop_length=hop,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0)
    log_mel = np.log1p(mel).astype(np.float32)   # (n_mels, T)
    return log_mel


def load_strip_2d(strip_path: Path, h_strip: int) -> np.ndarray:
    """Load grayscale strip resized to h_strip. Returns (h_strip, W) in [0,1].
    Score is INVERTED (1-img) to match their convention (white background → 0,
    black noteheads → 1).
    """
    img = Image.open(strip_path).convert("L")
    W, H = img.size
    if H != h_strip:
        img = img.resize((W, h_strip), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr   # invert: noteheads → 1, background → 0


def crop_score(strip: np.ndarray, cx: int, tile_width: int) -> np.ndarray:
    """Crop tile_width columns centered at cx from (H, W_full) strip.
    Zero-pads at edges. Returns (H, tile_width).
    """
    H, W_full = strip.shape
    x0 = max(0, cx - tile_width // 2)
    x0 = min(x0, max(0, W_full - tile_width))
    x1 = x0 + tile_width
    crop = strip[:, x0:min(x1, W_full)]
    if crop.shape[1] < tile_width:
        pad = np.zeros((H, tile_width - crop.shape[1]), dtype=np.float32)
        crop = np.concatenate([crop, pad], axis=1)
    return crop


def make_gt_mask(H: int, W: int, gt_width: int = 10, gt_height: int = None,
                 cx: int = None) -> np.ndarray:
    """Binary rectangle GT mask. cx defaults to W//2 (center). Returns (H, W) float32."""
    gt_height = gt_height or H // 2
    cx = cx if cx is not None else W // 2
    mask = np.zeros((H, W), dtype=np.float32)
    cy = H // 2
    y0 = max(0, cy - gt_height // 2)
    y1 = min(H, cy + gt_height // 2)
    x0 = max(0, cx - gt_width // 2)
    x1 = min(W, cx + gt_width // 2)
    mask[y0:y1, x0:x1] = 1.0
    return mask


class CPJKUDataset(Dataset):
    """Random-window dataset for training CPJKU network.

    Each __getitem__ samples a random frame t, returning:
      score_crop : (1, h_strip, tile_width) — 2D score crop centered at GT
      perf       : (1, n_mels, n_frames)    — spectrogram context (CBEncoder format)
      gt_mask    : (1, h_strip, tile_width) — binary GT rectangle at center
    """
    def __init__(self, processed_root: str, split: str,
                 tile_width: int = 512, h_strip: int = 128,
                 n_mels: int = 78, fps: int = 20, n_frames: int = 40,
                 gt_width: int = 10):
        self.root       = Path(processed_root)
        self.tile_width = tile_width
        self.h_strip    = h_strip
        self.n_mels     = n_mels
        self.fps        = fps
        self.n_frames   = n_frames
        self.gt_width   = gt_width

        splits = json.load(open(self.root / "splits.json"))
        self.piece_ids = splits[split]

    def __len__(self):
        return len(self.piece_ids)

    def _load_spec(self, pid, piece_dir):
        spec = _SPEC_CACHE.get(pid)
        if spec is None:
            spec = load_spec(piece_dir / "audio.wav", fps=self.fps, n_mels=self.n_mels)
            _SPEC_CACHE[pid] = spec
        return spec

    def __getitem__(self, idx: int) -> dict:
        pid = self.piece_ids[idx]
        piece_dir = self.root / pid

        ann   = json.load(open(piece_dir / "annotations.json"))
        notes = np.load(piece_dir / "noteheads.npz")
        strip_w = ann["image"]["width_px"]

        spec = self._load_spec(pid, piece_dir)  # (n_mels, T_spec)
        T_spec = spec.shape[-1]
        pad = self.n_frames    # their convention: pad n_frames zeros at start

        rng = np.random.default_rng()
        # Random frame t in [pad, T_spec)
        t = int(rng.integers(pad, max(pad + 1, T_spec)))
        t_sec = (t - pad) / self.fps   # real time (accounting for their padding)

        # GT strip_x at t_sec
        onset = notes["onset_sec"]
        gt_x = int(notes["strip_x"][int(np.argmin(np.abs(onset - t_sec)))]
                   if len(onset) else strip_w // 2)
        gt_x = int(np.clip(gt_x, 0, strip_w - 1))

        # Spectrogram context: 40 frames ending at t (their CBEncoder format)
        perf = spec[:, t - self.n_frames:t]   # (n_mels, n_frames)
        if perf.shape[-1] < self.n_frames:
            perf = np.pad(perf, ((0, 0), (self.n_frames - perf.shape[-1], 0)))

        # 2D score crop: GT placed at random position within the tile.
        # This forces the model to use audio FiLM to locate the note,
        # rather than learning the trivial "output center" shortcut.
        strip = load_strip_2d(piece_dir / "strip.png", self.h_strip)  # (H, W)
        margin = self.tile_width // 8   # 64px at tile_width=512
        local_gt_x = int(rng.integers(margin, self.tile_width - margin))
        x0 = int(np.clip(gt_x - local_gt_x, 0, max(0, strip.shape[1] - self.tile_width)))
        actual_local_gt_x = int(np.clip(gt_x - x0, 0, self.tile_width - 1))
        crop = strip[:, x0:x0 + self.tile_width]
        if crop.shape[1] < self.tile_width:
            crop = np.pad(crop, ((0, 0), (0, self.tile_width - crop.shape[1])))
        gt   = make_gt_mask(self.h_strip, self.tile_width, self.gt_width,
                            cx=actual_local_gt_x)

        # Their model expects (seq_len, bs, 1, H, W) for score and perf.
        # We return (1, H, W) tensors; train.py adds seq_len/bs dims.
        return {
            'score_crop':  torch.from_numpy(crop[np.newaxis]),       # (1, H, W)
            'perf':        torch.from_numpy(perf[np.newaxis]),       # (1, n_mels, n_frames)
            'gt_mask':     torch.from_numpy(gt[np.newaxis]),         # (1, H, W)
            'local_gt_x':  actual_local_gt_x,                       # int — GT x within crop
            'piece_id':    pid,
        }

    def compute_spec_stats(self) -> tuple:
        """Compute mean and std over training spectra for normalisation."""
        specs = []
        for pid in self.piece_ids:
            piece_dir = self.root / pid
            try:
                s = self._load_spec(pid, piece_dir)  # (n_mels, T)
                specs.append(s)
            except Exception:
                pass
        if not specs:
            return np.zeros(self.n_mels), np.ones(self.n_mels)
        cat = np.concatenate(specs, axis=-1)   # (n_mels, T_total)
        means = cat.mean(axis=1).astype(np.float32)
        stds  = cat.std(axis=1).astype(np.float32)
        stds[stds < 1e-6] = 1.0
        return means, stds
