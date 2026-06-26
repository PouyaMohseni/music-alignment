"""Full-piece dataset: one sample = one whole performance (cached embeddings).

Two storage backends, auto-detected by build_loader:
  - directory of <piece_id>.npz files  (FullSeqDataset)
  - tar shards + index.json            (FullSeqTarDataset; avoids inode blowup
    for big all-performances runs)

batch_size is effectively 1 (pieces vary in length), so collate returns the
single piece's tensors.
"""
from __future__ import annotations
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _arrays_to_sample(pid, z):
    return {
        "piece_id":   pid,
        "audio_emb":  torch.from_numpy(z["audio_emb"].astype(np.float32)),
        "tile_emb":   torch.from_numpy(z["tile_emb"].astype(np.float32)),
        "pos_tile":   torch.from_numpy(z["pos_tile"]),
        "pos_target": torch.from_numpy(z["pos_target"]),
        "valid_mask": torch.from_numpy(z["valid_mask"]),
        "eff_hz":     float(z["eff_hz"]),
        "px_per_sec": float(z["px_per_sec"]),
    }


class FullSeqDataset(Dataset):
    def __init__(self, emb_root: str, processed_root: str, split: str):
        self.emb_root = Path(emb_root)
        # aug embedding dirs carry their own splits.json; fall back to processed_root
        splits_path = self.emb_root / "splits.json"
        if not splits_path.exists():
            splits_path = Path(processed_root) / "splits.json"
        splits = json.load(open(splits_path))
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
        return _arrays_to_sample(pid, z)


class FullSeqTarDataset(Dataset):
    """Reads per-piece .npz blobs from tar shards via index.json.

    Tar handles are opened lazily per worker process (not picklable), so they
    survive DataLoader fork/spawn.
    """

    def __init__(self, tar_root: str, processed_root: str, split: str):
        self.tar_root = Path(tar_root)
        self.index = json.load(open(self.tar_root / "index.json"))   # piece_id -> shard
        splits_path = self.tar_root / "splits.json"
        if not splits_path.exists():
            splits_path = Path(processed_root) / "splits.json"
        splits = json.load(open(splits_path))
        if split not in splits:
            raise ValueError(f"split must be one of {list(splits)}")
        self.piece_ids = [p for p in splits[split] if p in self.index]
        if not self.piece_ids:
            raise ValueError(f"no tar-shard entries for split={split} in {tar_root}")
        self._tars: dict[str, tarfile.TarFile] = {}

    def __len__(self):
        return len(self.piece_ids)

    def _tar(self, shard):
        t = self._tars.get(shard)
        if t is None:
            t = tarfile.open(self.tar_root / shard, "r")
            self._tars[shard] = t
        return t

    def __getitem__(self, idx):
        pid = self.piece_ids[idx]
        tar = self._tar(self.index[pid])
        raw = tar.extractfile(f"{pid}.npz").read()
        z = np.load(io.BytesIO(raw))
        return _arrays_to_sample(pid, z)


def _collate(batch):
    return batch[0]   # batch_size always 1


def build_loader(emb_root, processed_root, split, shuffle, num_workers=2):
    # Auto-detect: tar shards (index.json present) vs directory of .npz files
    if (Path(emb_root) / "index.json").exists():
        ds = FullSeqTarDataset(emb_root, processed_root, split)
    else:
        ds = FullSeqDataset(emb_root, processed_root, split)
    return DataLoader(ds, batch_size=1, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=_collate,
                      persistent_workers=num_workers > 0)
