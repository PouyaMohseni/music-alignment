"""v11-mert-finetune: like v11_cpjku_fullstrip, but the audio side is a LIVE,
fine-tunable MERT-v1-95M (mymodel/v10_mert_unet/mert_live.py) instead of a
static spectrogram (real-madmom or mel-approximation). Score/GT-mask logic
is untouched and reused verbatim from v11_cpjku_fullstrip -- only the audio
representation changes. Per-frame embeddings are computed on-the-fly inside
the BPTT loop (train.py), not here -- this module only loads and holds the
raw waveform per piece.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from mymodel.v11_cpjku_fullstrip.data import load_strip_scaled
from mymodel.v10_mert_unet.mert_live import MERT_SR


def load_raw_audio(wav_path: Path, sr: int = MERT_SR) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    return y.astype(np.float32)


def load_piece(piece_dir: Path, h_strip: int, w_scale: int, fps: int = 20) -> dict | None:
    """Load one piece. Returns dict with all data needed for BPTT training."""
    try:
        notes = np.load(piece_dir / 'noteheads.npz')
    except Exception:
        return None

    try:
        audio = load_raw_audio(piece_dir / 'audio.wav')
    except Exception:
        return None

    try:
        score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
    except Exception:
        return None

    H, W_sc = score.shape
    T = int(np.floor(len(audio) / MERT_SR * fps))
    if T <= 0:
        return None

    onset_sec   = notes['onset_sec'].astype(np.float64)
    strip_x_raw = notes['strip_x'].astype(np.float32)
    onset_frames = np.round(onset_sec * fps).astype(np.int64)
    onset_frames = np.clip(onset_frames, 0, T - 1)

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
        'audio':        audio,          # (n_samples,) float32 @ MERT_SR (24kHz)
        'strip_x_sc':   strip_x_sc,     # (T,) GT x in scaled strip coords
        'onset_frames': onset_frames,   # (N,) for eval
        'strip_x_raw':  strip_x_raw,    # (N,) original px, for eval
        'W_sc':         W_sc,
        'H':            H,
        'T':            T,
        'pid':          piece_dir.name,
    }


class FullStripAudioDataset:
    """Pre-loads all pieces (score + raw waveform) for BPTT training."""

    def __init__(self, processed_root: str, split: str,
                 h_strip: int = 128, w_scale: int = 4, fps: int = 20):
        self.root    = Path(processed_root)
        self.h_strip = h_strip
        self.w_scale = w_scale
        self.fps     = fps

        splits = json.load(open(self.root / 'splits.json'))
        piece_ids = splits[split]

        print(f'Loading {len(piece_ids)} pieces ({split})...', flush=True)
        self.pieces = []
        for pid in piece_ids:
            d = load_piece(self.root / pid, h_strip, w_scale, fps)
            if d is not None:
                self.pieces.append(d)
            else:
                print(f'  SKIP {pid}', flush=True)
        print(f'  Loaded {len(self.pieces)}/{len(piece_ids)} pieces.', flush=True)

    def __len__(self) -> int:
        return len(self.pieces)

    def __getitem__(self, idx: int) -> dict:
        return self.pieces[idx]
