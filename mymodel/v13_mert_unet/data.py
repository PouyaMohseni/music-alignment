"""v13/v14/v15: Full-strip dataset using pre-computed MERT features (20fps, 768-dim).

Features at data/MSMD/mert_emb/<pid>.npy — float16, shape (T, 768).
All three variants share this dataset; only the audio encoder differs.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import interp1d


def load_strip_scaled(strip_path: Path, h: int, w_scale: int) -> np.ndarray:
    img = Image.open(strip_path).convert('L')
    W_sc = max(1, img.width // w_scale)
    img = img.resize((W_sc, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr  # invert: noteheads → 1


def make_gt_mask(H: int, W: int, cx: int, gt_width: int = 10) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.float32)
    cy = H // 2
    gt_height = H // 2
    y0, y1 = max(0, cy - gt_height // 2), min(H, cy + gt_height // 2)
    x0, x1 = max(0, cx - gt_width // 2), min(W, cx + gt_width // 2)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def load_piece(piece_dir: Path, emb_root: str, h_strip: int, w_scale: int,
               fps: int = 20) -> dict | None:
    try:
        notes = np.load(piece_dir / 'noteheads.npz')
    except Exception:
        return None

    emb_path = Path(emb_root) / f'{piece_dir.name}.npy'
    if not emb_path.exists():
        return None
    try:
        feats = np.load(str(emb_path)).astype(np.float32)  # (T, 768)
    except Exception:
        return None

    try:
        score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
    except Exception:
        return None

    H, W_sc = score.shape
    T = feats.shape[0]

    onset_sec   = notes['onset_sec'].astype(np.float64)
    strip_x_raw = notes['strip_x'].astype(np.float32)
    onset_frames = np.clip(np.round(onset_sec * fps).astype(np.int64), 0, T - 1)

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
        'score':        score,        # (H, W_sc) float32
        'feats':        feats,        # (T, 768) MERT at 20fps
        'strip_x_sc':   strip_x_sc,   # (T,)
        'onset_frames': onset_frames, # (N,)
        'strip_x_raw':  strip_x_raw,  # (N,)
        'W_sc': W_sc, 'H': H, 'T': T,
        'pid':  piece_dir.name,
    }


class FullStripDataset:
    def __init__(self, processed_root: str, emb_root: str, split: str,
                 h_strip: int = 128, w_scale: int = 4, fps: int = 20):
        self.root = Path(processed_root)
        splits = json.load(open(self.root / 'splits.json'))
        piece_ids = splits[split]

        print(f'Loading {len(piece_ids)} pieces ({split})...', flush=True)
        self.pieces = []
        for pid in piece_ids:
            d = load_piece(self.root / pid, emb_root, h_strip, w_scale, fps)
            if d is not None:
                self.pieces.append(d)
            else:
                print(f'  SKIP {pid}', flush=True)
        print(f'  Loaded {len(self.pieces)}/{len(piece_ids)} pieces.', flush=True)

    def __len__(self):  return len(self.pieces)
    def __getitem__(self, i): return self.pieces[i]
