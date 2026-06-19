"""v4 data: v3 precomputed embeddings + on-the-fly 88-key pitch targets.

Wraps the v3 embedding dataset (dir-of-npz or tar-shards, auto-detected) and adds
audio_pitch_target (T,88) and score_pitch_target (N,88) built from the piece's
noteheads.npz. No re-precompute needed — targets are cheap to build per item.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..v3_fullseq.data import FullSeqDataset, FullSeqTarDataset
from .pitchroll import audio_pitchroll, score_pitchroll


class PitchFusedDataset(Dataset):
    def __init__(self, emb_root, processed_root, split, tile_size=224):
        if (Path(emb_root) / "index.json").exists():
            self.base = FullSeqTarDataset(emb_root, processed_root, split)
        else:
            self.base = FullSeqDataset(emb_root, processed_root, split)
        self.processed_root = Path(processed_root)
        self.tile_size = tile_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        pid = s["piece_id"]
        pdir = self.processed_root / pid
        notes = np.load(pdir / "noteheads.npz")
        ann = json.load(open(pdir / "annotations.json"))
        strip_w = float(ann["image"]["width_px"])

        T = s["audio_emb"].shape[0]
        N = s["tile_emb"].shape[0]
        eff_hz = s["eff_hz"]
        tile_centers_px = s["pos_tile"].numpy() * strip_w

        a_tgt = audio_pitchroll(notes["onset_sec"], notes["midi_offset_sec"],
                                notes["midi_pitch"], T, eff_hz)
        s_tgt = score_pitchroll(notes["strip_x"], notes["midi_pitch"],
                                tile_centers_px, self.tile_size)
        s["audio_pitch_target"] = torch.from_numpy(a_tgt)     # (T,88)
        s["score_pitch_target"] = torch.from_numpy(s_tgt)     # (N,88)
        return s


def _collate(batch):
    return batch[0]


def build_loader(emb_root, processed_root, split, shuffle, num_workers=2, tile_size=224):
    ds = PitchFusedDataset(emb_root, processed_root, split, tile_size=tile_size)
    return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=_collate, persistent_workers=num_workers > 0)
