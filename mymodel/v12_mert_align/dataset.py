"""
MSMD dataset for v12 MERT alignment.

Each item is one piece. Audio loaded as raw waveform at 24kHz.
Score strip sliced into overlapping 80px-wide columns (stride 40px).
Ground truth: (onset_sec, strip_x_px) pairs from noteheads.npz.
"""
import json, os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import soundfile as sf
import torchvision.transforms as T

COL_WIDTH  = 80
COL_STRIDE = 40
STRIP_H    = 224
AUDIO_SR   = 24000
MERT_HZ    = 75   # MERT output frame rate


def strip_to_columns(strip: np.ndarray) -> torch.Tensor:
    """
    strip: (H, W, 3) uint8
    Returns: (N_cols, 3, 224, 224) float32 in [0,1]
    """
    H, W, _ = strip.shape
    assert H == STRIP_H
    transform = T.Compose([
        T.ToTensor(),          # -> (3, H, col_w)
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    cols = []
    start = 0
    while start + COL_WIDTH <= W:
        patch = strip[:, start:start + COL_WIDTH, :]     # (H, 80, 3)
        col_img = Image.fromarray(patch)
        cols.append(transform(col_img))
        start += COL_STRIDE
    return torch.stack(cols)   # (N_cols, 3, 224, 224)


def col_centers(N_cols: int) -> np.ndarray:
    """x_px of the centre of each column."""
    return np.array([i * COL_STRIDE + COL_WIDTH // 2 for i in range(N_cols)],
                    dtype=np.float32)


def x_to_col(x_px: np.ndarray, N_cols: int) -> np.ndarray:
    """Map strip x-pixels to nearest column index."""
    idx = np.round((x_px - COL_WIDTH // 2) / COL_STRIDE).astype(int)
    return np.clip(idx, 0, N_cols - 1)


class MSMDPiece:
    """Holds one fully-loaded MSMD piece."""
    def __init__(self, piece_dir: str):
        # Audio
        wav, sr = sf.read(os.path.join(piece_dir, 'audio.wav'), dtype='float32')
        assert sr == AUDIO_SR, f"Expected 24kHz, got {sr}"
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        self.wav = torch.from_numpy(wav)           # (T_samples,)

        # Score strip -> columns
        strip_img = Image.open(os.path.join(piece_dir, 'strip.png')).convert('RGB')
        strip = np.array(strip_img)                # (224, W, 3)
        self.strip_W = strip.shape[1]
        self.score_cols = strip_to_columns(strip)  # (N_cols, 3, 224, 224)
        self.N_cols = self.score_cols.shape[0]
        self.centers = col_centers(self.N_cols)    # (N_cols,) x_px

        # Annotations
        ann = np.load(os.path.join(piece_dir, 'noteheads.npz'))
        self.onset_sec  = ann['onset_sec'].astype(np.float32)   # (N,)
        self.strip_x    = ann['strip_x'].astype(np.float32)     # (N,)

        # Convert to frame / col indices
        self.onset_frames = np.round(self.onset_sec * MERT_HZ).astype(int)
        self.onset_cols   = x_to_col(self.strip_x, self.N_cols)


class MSMDDataset(Dataset):
    """One item = one piece (variable length)."""
    def __init__(self, split: str, data_root: str = 'data/MSMD/processed'):
        splits = json.load(open(os.path.join(data_root, 'splits.json')))
        self.pieces = []
        print(f"Loading {split} set ({len(splits[split])} pieces)...")
        for pid in splits[split]:
            pdir = os.path.join(data_root, pid)
            if os.path.isdir(pdir):
                try:
                    self.pieces.append(MSMDPiece(pdir))
                except Exception as e:
                    print(f"  skip {pid}: {e}")
        print(f"  Loaded {len(self.pieces)} pieces.")

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        return self.pieces[idx]
