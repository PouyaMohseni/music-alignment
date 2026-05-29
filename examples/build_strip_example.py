"""Run msmd_prep.strip on Bach BWV 827-2 (3 pages, 14 systems) and save outputs.

From the repo root:
    python examples/build_strip_example.py
"""
from __future__ import annotations
import json, os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from msmd_prep.strip import build_strip

PIECE_ID = "BachJS__BWV827__BWV-827-2"
SCORE_DIR = os.path.join(
    os.path.dirname(THIS_DIR),
    "data", "MSMD", "msmd_aug_v1-1_no-audio", PIECE_ID,
    "scores", f"{PIECE_ID}_ly",
)
OUT_DIR = THIS_DIR

strip, mapping = build_strip(SCORE_DIR)

strip_path   = os.path.join(OUT_DIR, f"{PIECE_ID}_strip.png")
mapping_path = os.path.join(OUT_DIR, f"{PIECE_ID}_mapping.json")
strip.save(strip_path)
with open(mapping_path, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"strip   -> {strip_path}  size={strip.size}")
print(f"mapping -> {mapping_path}  systems={len(mapping)}")
