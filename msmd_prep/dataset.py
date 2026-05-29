"""Reference loader. Framework-agnostic; converts cleanly to torch / HF datasets."""
from __future__ import annotations
import json, os
import numpy as np


class MSMDAlignmentDataset:
    """Reads manifest.jsonl produced by msmd_prep.manifest.build_manifest.

    Each sample is a dict:
        piece_id:   str
        image_path: str (absolute)
        audio_path: str | None
        midi_path:  str
        annotations: dict   (the JSON sidecar, parsed)
        noteheads:  dict[str, np.ndarray]   (columnar arrays from .npz)

    Plug into PyTorch via:
        class TorchView(torch.utils.data.Dataset):
            def __init__(self, root, split):
                self.inner = MSMDAlignmentDataset(root, split=split)
            def __len__(self): return len(self.inner)
            def __getitem__(self, i): return self.inner[i]    # apply your transforms here
    """

    def __init__(self, root: str, split: str | None = None):
        self.root = root
        manifest_path = os.path.join(root, "manifest.jsonl")
        with open(manifest_path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if split:
            self.rows = [r for r in self.rows if r["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def _abs(self, rel: str | None) -> str | None:
        return os.path.join(self.root, rel) if rel else None

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        with open(self._abs(r["annotations"])) as f:
            ann = json.load(f)
        npz = np.load(self._abs(r["noteheads"]))
        return {
            "piece_id":    r["piece_id"],
            "image_path":  self._abs(r["image"]),
            "audio_path":  self._abs(r["audio"]),
            "midi_path":   self._abs(r["midi"]),
            "annotations": ann,
            "noteheads":   {k: npz[k] for k in npz.files},
        }
