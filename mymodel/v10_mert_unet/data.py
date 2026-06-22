"""v10 data loader — precomputed MERT embeddings + 2D score strip crops.

Each training sample: random frame t → MERT embedding (768,) + 2D score crop.
Reuses v9's strip/crop/GT helpers; only the audio representation changes.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..v9_cpjku.data import load_strip_2d, crop_score, make_gt_mask

_MERT_CACHE: dict = {}


class MERTDataset(Dataset):
    def __init__(self, processed_root, mert_emb_root, split,
                 tile_width=512, h_strip=128, fps=20, gt_width=10):
        self.root      = Path(processed_root)
        self.emb_root  = Path(mert_emb_root)
        self.tile_width = tile_width
        self.h_strip    = h_strip
        self.fps        = fps
        self.gt_width   = gt_width

        splits = json.load(open(self.root / 'splits.json'))
        self.piece_ids = [p for p in splits[split]
                          if (self.emb_root / f'{p}.npy').exists()]
        if not self.piece_ids:
            raise ValueError(f'No MERT .npy files in {mert_emb_root} for split={split}. '
                             'Run: python -m mymodel.v10_mert_unet.precompute')

    def __len__(self):
        return len(self.piece_ids)

    def _load_emb(self, pid):
        emb = _MERT_CACHE.get(pid)
        if emb is None:
            emb = np.load(self.emb_root / f'{pid}.npy').astype(np.float32)  # (T, 768)
            _MERT_CACHE[pid] = emb
        return emb

    def __getitem__(self, idx):
        pid = self.piece_ids[idx]
        piece_dir = self.root / pid

        ann   = json.load(open(piece_dir / 'annotations.json'))
        notes = np.load(piece_dir / 'noteheads.npz')
        strip_w = ann['image']['width_px']

        mert_emb = self._load_emb(pid)   # (T, 768)
        T = mert_emb.shape[0]

        rng = np.random.default_rng()
        t = int(rng.integers(1, max(2, T)))
        t_sec = t / self.fps

        onset = notes['onset_sec']
        gt_x = int(notes['strip_x'][int(np.argmin(np.abs(onset - t_sec)))]
                   if len(onset) else strip_w // 2)
        gt_x = int(np.clip(gt_x, 0, strip_w - 1))

        perf = mert_emb[t]   # (768,)

        strip = load_strip_2d(piece_dir / 'strip.png', self.h_strip)
        crop  = crop_score(strip, gt_x, self.tile_width)
        gt    = make_gt_mask(self.h_strip, self.tile_width, self.gt_width)

        # Reshape perf to (1, 768, 1) — mirrors CBEncoder's (1, n_mels, n_frames) format.
        # train.py unsqueezes batch dim → (B, 1, 768, 1), then seq_len → (1, B, 1, 768, 1).
        return {
            'score_crop': torch.from_numpy(crop[np.newaxis]),                      # (1, H, W)
            'perf':       torch.from_numpy(perf[np.newaxis, :, np.newaxis]),       # (1, 768, 1)
            'gt_mask':    torch.from_numpy(gt[np.newaxis]),                        # (1, H, W)
            'piece_id':   pid,
        }

    def compute_spec_stats(self):
        """No-op: MERT embeddings are already normalized. Returns dummy zeros/ones."""
        return np.zeros(768, dtype=np.float32), np.ones(768, dtype=np.float32)
