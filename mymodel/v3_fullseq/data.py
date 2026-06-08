"""Full-piece dataset: one sample = one whole performance (cached embeddings).

Reads the .npz files produced by precompute.py. Each __getitem__ returns the
full sequence for a piece; batch_size is effectively 1 (pieces vary in length),
so we use a trivial collate that returns the single piece's tensors.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FullSeqDataset(Dataset):
    def __init__(self, emb_root: str, processed_root: str, split: str):
        self.emb_root = Path(emb_root)
        splits = json.load(open(Path(processed_root) / "splits.json"))
        if split not in splits:
            raise ValueError(f"split must be one of {list(splits)}")
        # keep only pieces whose embedding file exists
        self.piece_ids = [p for p in splits[split]
                          if (self.emb_root / f"{p}.npz").exists()]
        if not self.piece_ids:
            raise ValueError(f"no embedding files for split={split} in {emb_root}")

    def __len__(self):
        return len(self.piece_ids)

    def __getitem__(self, idx):
        pid = self.piece_ids[idx]
        z = np.load(self.emb_root / f"{pid}.npz")
        return {
            "piece_id":   pid,
            "audio_emb":  torch.from_numpy(z["audio_emb"].astype(np.float32)),   # (T, Da)
            "tile_emb":   torch.from_numpy(z["tile_emb"].astype(np.float32)),    # (N, Di)
            "pos_tile":   torch.from_numpy(z["pos_tile"]),                       # (N,)
            "pos_target": torch.from_numpy(z["pos_target"]),                     # (T,)
            "valid_mask": torch.from_numpy(z["valid_mask"]),                     # (T,)
            "eff_hz":     float(z["eff_hz"]),
            "px_per_sec": float(z["px_per_sec"]),
        }


def _collate(batch):
    return batch[0]   # batch_size always 1


def build_loader(emb_root, processed_root, split, shuffle, num_workers=2):
    ds = FullSeqDataset(emb_root, processed_root, split)
    return DataLoader(ds, batch_size=1, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=_collate,
                      persistent_workers=num_workers > 0)
