"""Build a HF-datasets-friendly manifest.jsonl + splits.json from per-piece dirs."""
from __future__ import annotations
import json, os

from .schema import PER_PIECE_FILES


def build_manifest(dataset_root: str, splits: dict[str, list[str]] | None = None):
    """Walk dataset_root for piece subdirs and write manifest.jsonl (+ splits.json).

    Each manifest row has:
        piece_id, image, audio, midi, annotations, noteheads, split

    All path fields are relative to dataset_root, so HF datasets can load them.
    Pieces without an audio.wav set audio=None.
    """
    piece_ids = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
        and os.path.exists(os.path.join(dataset_root, d, PER_PIECE_FILES["annotations"]))
    )
    split_lookup = {}
    if splits:
        for split, ids in splits.items():
            for pid in ids:
                split_lookup[pid] = split

    manifest_path = os.path.join(dataset_root, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for pid in piece_ids:
            piece_dir = os.path.join(dataset_root, pid)
            audio_rel = f"{pid}/{PER_PIECE_FILES['audio']}"
            has_audio = os.path.exists(os.path.join(dataset_root, audio_rel))
            row = {
                "piece_id":    pid,
                "image":       f"{pid}/{PER_PIECE_FILES['image']}",
                "audio":       audio_rel if has_audio else None,
                "midi":        f"{pid}/{PER_PIECE_FILES['midi']}",
                "annotations": f"{pid}/{PER_PIECE_FILES['annotations']}",
                "noteheads":   f"{pid}/{PER_PIECE_FILES['noteheads']}",
                "split":       split_lookup.get(pid, "train"),
            }
            f.write(json.dumps(row) + "\n")

    if splits:
        with open(os.path.join(dataset_root, "splits.json"), "w") as f:
            json.dump(splits, f, indent=2)

    return manifest_path
