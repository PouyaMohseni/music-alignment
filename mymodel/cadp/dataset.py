"""CADP dataset — loads pre-computed MERT + DINOv2 features for alignment."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np


class CADPDataset:
    """Per-piece dataset for CADP training.

    Loads:
      - MERT features: (T, 768) float16, at 20fps
      - DINOv2 column features: (N_cols, 16, 768) float16
      - GT onset annotations: onset_sec, strip_x

    col_stride_px: pixel stride between score columns (default 40)
    col_w_px:      pixel width of each column window (default 80)
    """

    def __init__(self, processed_root: str, mert_root: str, dinov2_root: str,
                 split: str, col_w_px: int = 80, col_stride_px: int = 40,
                 fps: float = 20.0):
        self.proc = Path(processed_root)
        self.mert = Path(mert_root)
        self.d2   = Path(dinov2_root)
        self.col_w = col_w_px
        self.col_stride = col_stride_px
        self.fps = fps

        splits = json.load(open(self.proc / 'splits.json'))
        self.piece_ids = splits[split]

    def load_piece(self, pid: str) -> Optional[dict]:
        mert_path = self.mert / f'{pid}.npy'
        d2_path   = self.d2   / f'{pid}.npy'
        ann_path  = self.proc / pid / 'annotations.json'
        nh_path   = self.proc / pid / 'noteheads.npz'

        if not mert_path.exists() or not d2_path.exists():
            return None

        mert_feats = np.load(str(mert_path)).astype(np.float32)   # (T, 768)
        d2_feats   = np.load(str(d2_path)).astype(np.float32)     # (N_cols, 16, 768)
        ann        = json.load(open(ann_path))
        nh         = np.load(str(nh_path))

        onset_sec  = nh['onset_sec'].astype(np.float32)
        strip_x    = nh['strip_x'].astype(np.float32)
        strip_w    = float(ann['image']['width_px'])
        dur        = float(ann['audio']['duration_sec'])

        # Number of columns: should match d2_feats.shape[0]
        # Compute expected N_cols from strip_w
        import json as _json
        from PIL import Image
        strip_path = self.proc / pid / 'strip.png'
        if strip_path.exists():
            img = Image.open(strip_path)
            W_orig, H_orig = img.size
            h_target = 224
            if H_orig != h_target:
                W_orig = int(W_orig * h_target / H_orig)
            n_cols_expected = max(1, (W_orig - self.col_w) // self.col_stride + 1)
        else:
            n_cols_expected = d2_feats.shape[0]

        # Map onset_sec → frame index
        frame_idx = np.clip(np.round(onset_sec * self.fps).astype(np.int64),
                            0, mert_feats.shape[0] - 1)

        # Map strip_x → column index
        # Column i covers pixels [i*stride, i*stride + col_w)
        # Center of column i = i*stride + col_w/2
        col_idx = np.clip(
            np.round((strip_x - self.col_w / 2.0) / self.col_stride).astype(np.int64),
            0, d2_feats.shape[0] - 1
        )

        return {
            'pid':        pid,
            'mert_feats': mert_feats,     # (T, 768)
            'd2_feats':   d2_feats,        # (N_cols, 16, 768)
            'onset_sec':  onset_sec,
            'strip_x':    strip_x,
            'frame_idx':  frame_idx,       # GT frame index per note
            'col_idx':    col_idx,         # GT column index per note
            'strip_w':    strip_w,
            'dur':        dur,
            'fps':        self.fps,
            'col_stride': float(self.col_stride),
        }

    def __len__(self):
        return len(self.piece_ids)

    def __getitem__(self, idx: int) -> Optional[dict]:
        return self.load_piece(self.piece_ids[idx])
