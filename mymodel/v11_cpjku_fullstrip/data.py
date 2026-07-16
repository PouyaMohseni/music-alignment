"""v11: Full-strip dataset for BPTT training.

Each piece provides its FULL strip (scaled) as the score image for every frame.
GT position is the TRUE note x-coordinate per frame — it varies naturally
as the piece progresses, so the model must use audio to locate the note.

Key difference from v9: no cropping, no always-centred GT.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import interp1d


def load_spec(wav_path: Path, sr: int = 22050, fps: int = 20,
              n_mels: int = 78, fmin: float = 60.0, fmax: float = 6000.0) -> np.ndarray:
    """Mel-spectrogram approximation. Kept as the fallback path -- prefer
    load_spec_madmom (real madmom LogarithmicFilterbank, matches what the
    CBEncoder architecture was actually designed/tuned around) whenever a
    cache is available. See mymodel/cpjku_adapter/precompute_madmom_specs.py."""
    import librosa
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    hop = int(sr / fps)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=hop,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0)
    return np.log1p(mel).astype(np.float32)  # (n_mels, T)


def load_spec_madmom(pid: str, cpjku_fmt_root: Path, pad: int = 40) -> np.ndarray | None:
    """Load the real-madmom spectrogram cached by precompute_madmom_specs.py
    (data/MSMD/cpjku_fmt/spec_madmom/<pid>.npy), de-padded to match this
    module's frame convention (T = audio_len/hop, no leading context pad --
    that pad exists only for eval_official.py's rolling-window inference).
    Returns None if no cache exists for this piece."""
    path = Path(cpjku_fmt_root) / 'spec_madmom' / f'{pid}.npy'
    if not path.exists():
        return None
    spec = np.load(path)
    return spec[:, pad:].astype(np.float32)


def load_strip_scaled(strip_path: Path, h: int, w_scale: int) -> np.ndarray:
    """Load + downscale strip. Returns (h, W//w_scale) float32, inverted (noteheads=1)."""
    img = Image.open(strip_path).convert('L')
    W_orig, H_orig = img.size
    W_sc = max(1, W_orig // w_scale)
    img = img.resize((W_sc, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr  # invert: noteheads → 1, background → 0


def make_gt_mask(H: int, W: int, cx: int, gt_width: int = 10) -> np.ndarray:
    """Binary rectangle at cx, centred vertically. Returns (H, W) float32."""
    gt_height = H // 2
    mask = np.zeros((H, W), dtype=np.float32)
    cy = H // 2
    y0 = max(0, cy - gt_height // 2)
    y1 = min(H, cy + gt_height // 2)
    x0 = max(0, cx - gt_width // 2)
    x1 = min(W, cx + gt_width // 2)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def load_piece(piece_dir: Path, h_strip: int, w_scale: int,
               n_mels: int = 78, fps: int = 20,
               cpjku_fmt_root: Path | None = None) -> dict | None:
    """Load one piece. Returns dict with all data needed for BPTT training."""
    try:
        notes = np.load(piece_dir / 'noteheads.npz')
    except Exception:
        return None

    spec = None
    if cpjku_fmt_root is not None:
        try:
            spec = load_spec_madmom(piece_dir.name, cpjku_fmt_root)
        except Exception:
            spec = None
        if spec is None:
            print(f'WARNING: no cached real-madmom spectrogram for {piece_dir.name} '
                  f'-- falling back to mel-spectrogram approximation.', flush=True)
    if spec is None:
        try:
            spec = load_spec(piece_dir / 'audio.wav', fps=fps, n_mels=n_mels)
        except Exception:
            return None

    try:
        score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
    except Exception:
        return None

    H, W_sc = score.shape
    T = spec.shape[-1]

    onset_sec   = notes['onset_sec'].astype(np.float64)
    strip_x_raw = notes['strip_x'].astype(np.float32)
    onset_frames = np.round(onset_sec * fps).astype(np.int64)
    onset_frames = np.clip(onset_frames, 0, T - 1)

    # Per-frame GT x in scaled strip space via piecewise-constant interpolation
    t_sec_all = np.arange(T, dtype=np.float64) / fps
    if len(onset_sec) == 0:
        strip_x_sc = np.full(T, W_sc // 2, dtype=np.float32)
    else:
        interp = interp1d(
            onset_sec, strip_x_raw / w_scale,
            kind='previous', bounds_error=False,
            fill_value=(strip_x_raw[0] / w_scale, strip_x_raw[-1] / w_scale))
        strip_x_sc = interp(t_sec_all).astype(np.float32)
    strip_x_sc = np.clip(strip_x_sc, 0, W_sc - 1)

    return {
        'score':        score,          # (H, W_sc) float32
        'spec':         spec,           # (n_mels, T) float32
        'strip_x_sc':   strip_x_sc,     # (T,) GT x in scaled strip coords
        'onset_frames': onset_frames,   # (N,) for eval
        'strip_x_raw':  strip_x_raw,    # (N,) original px, for eval
        'W_sc':         W_sc,
        'H':            H,
        'T':            T,
        'pid':          piece_dir.name,
    }


class FullStripDataset:
    """Pre-loads all pieces for BPTT training. Each item is a complete piece."""

    def __init__(self, processed_root: str, split: str,
                 h_strip: int = 128, w_scale: int = 4,
                 n_mels: int = 78, fps: int = 20,
                 cpjku_fmt_root: str | None = None):
        self.root    = Path(processed_root)
        self.h_strip = h_strip
        self.w_scale = w_scale
        self.n_mels  = n_mels
        self.fps     = fps
        self.cpjku_fmt_root = Path(cpjku_fmt_root) if cpjku_fmt_root else None

        splits = json.load(open(self.root / 'splits.json'))
        piece_ids = splits[split]

        print(f'Loading {len(piece_ids)} pieces ({split})...', flush=True)
        self.pieces = []
        for pid in piece_ids:
            d = load_piece(self.root / pid, h_strip, w_scale, n_mels, fps,
                            cpjku_fmt_root=self.cpjku_fmt_root)
            if d is not None:
                self.pieces.append(d)
            else:
                print(f'  SKIP {pid}', flush=True)
        print(f'  Loaded {len(self.pieces)}/{len(piece_ids)} pieces.', flush=True)

    def __len__(self) -> int:
        return len(self.pieces)

    def __getitem__(self, idx: int) -> dict:
        return self.pieces[idx]

    def compute_spec_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std per mel band across all training audio."""
        specs = [p['spec'] for p in self.pieces]
        cat = np.concatenate(specs, axis=-1)  # (n_mels, T_total)
        means = cat.mean(axis=1).astype(np.float32)
        stds  = cat.std(axis=1).astype(np.float32)
        stds[stds < 1e-6] = 1.0
        return means, stds
