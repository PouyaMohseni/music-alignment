"""Produce a DL-ready per-piece directory for the example MSMD piece.

From the repo root:
    python examples/build_piece_example.py

Output layout:
    examples/dataset/
        manifest.jsonl
        splits.json
        BachJS__BWV827__BWV-827-2/
            strip.png
            score.midi
            annotations.json
            noteheads.npz
            (audio.wav -- to be produced by Stage 2 / synth.py)
"""
from __future__ import annotations
import os, sys, json
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from msmd_prep.piece import build_piece
from msmd_prep.manifest import build_manifest
from msmd_prep.dataset import MSMDAlignmentDataset

PIECE_ID    = "BachJS__BWV772__bach-invention-01"
MSMD_ROOT   = os.path.join(REPO_ROOT, "data", "MSMD", "msmd_aug_v1-1_no-audio")
OUT_ROOT    = os.path.join(THIS_DIR, "dataset")

piece_dir = os.path.join(MSMD_ROOT, PIECE_ID)
out_dir   = os.path.join(OUT_ROOT, PIECE_ID)

info = build_piece(piece_dir, out_dir)
for k, v in info.items():
    print(f"  {k:18} {v}")

manifest_path = build_manifest(OUT_ROOT, splits={"train": [PIECE_ID], "val": [], "test": []})
print(f"\nmanifest -> {manifest_path}")

# Sanity load through the reference Dataset.
ds = MSMDAlignmentDataset(OUT_ROOT, split="train")
sample = ds[0]
print(f"\nDataset loaded {len(ds)} piece(s).  First sample:")
print(f"  piece_id     {sample['piece_id']}")
print(f"  image_path   {sample['image_path']}")
print(f"  audio_path   {sample['audio_path']}")
print(f"  midi_path    {sample['midi_path']}")
print(f"  noteheads    {dict((k, v.shape) for k, v in sample['noteheads'].items())}")
print(f"  duration_sec {sample['annotations']['audio']['duration_sec']:.3f}")
print(f"  first 3 noteheads:")
nh = sample['noteheads']
for i in range(3):
    print(f"    onset={nh['onset_sec'][i]:6.3f}  strip_x={nh['strip_x'][i]:5d}  "
          f"pitch={nh['midi_pitch'][i]:3d}  system={nh['system_idx'][i]:2d}  "
          f"measure={nh['measure_idx'][i]:3d}")
