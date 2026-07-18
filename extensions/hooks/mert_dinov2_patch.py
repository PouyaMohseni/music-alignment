"""Merged load_piece patch for MERT+DINOv2 (low-risk bottleneck-injection
variant): B1a's frozen-MERT load_piece patch (mert_patch.py) PLUS a
constant-per-page DINOv2 CLS embedding, concatenated onto every frame of
the MERT audio embedding.

Why concatenation instead of touching network.py/iterate_dataset: the
score image (and hence its DINOv2 embedding) is FIXED for the whole piece
-- ScoreAudioDataset.__getitem__ uses the same score tensor for every
frame -- so there is no need to plumb a new argument through
ConditionalUNet.forward's call signature (which every caller in this
codebase currently invokes as network(score=..., perf=..., hidden=...),
fixed positional/keyword args). Instead, broadcast the SAME 768-dim DINOv2
CLS vector across all T frames and concatenate it onto MERT's per-frame
768-dim vector, giving a (1536, T) 'spec' that MERTDINOv2Projector (a new
audio_encoder registered alongside this patch) knows how to split back
into its audio and visual halves. Zero changes to network.py, dataset.py's
call sites, or iterate_dataset -- everything downstream is unchanged.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from extensions.hooks.mert_patch import _patched_load_piece as _mert_load_piece


def _load_dinov2_cls(dinov2_root: str, piece_name: str) -> np.ndarray:
    """piece_name is the native page id, e.g. <pid>_page_N -- DINOv2 embeddings
    are keyed by page only (no tempo suffix, since the score image doesn't
    change with tempo -- see scripts/precompute_dinov2_native_pages.py)."""
    path = Path(dinov2_root) / 'cls' / f'{piece_name}.npy'
    return np.load(path).astype(np.float32)   # (768,)


def _patched_load_piece_mert_dinov2(params):
    """Runs mert_patch's own _patched_load_piece unchanged (gives us a
    correct (768, T) MERT spec + onsets + interpol_fnc/c2o/add_per_staff
    per tempo_factor), then concatenates the constant DINOv2 CLS embedding
    onto every frame of each tempo variant's spec."""
    import os
    dinov2_root = params['dinov2_root']
    piece_name = params['piece_name']

    i, score, name, performances = _mert_load_piece(params)

    dinov2_cls = _load_dinov2_cls(dinov2_root, piece_name)   # (768,)

    for tempo_factor, perf in performances.items():
        spec = perf['spec']                     # (768, T) MERT, already padded
        T = spec.shape[-1]
        visual_bcast = np.repeat(dinov2_cls[:, None], T, axis=1)   # (768, T)
        perf['spec'] = np.concatenate([spec, visual_bcast], axis=0)   # (1536, T)

    return i, score, name, performances


def patch_mert_dinov2_pipeline(path_to_emb_root: dict[str, str], dinov2_root: str):
    """Call once, before load_dataset()/ConditionalUNet(). path_to_emb_root
    is the same MERT audio-embedding mapping mert_patch.py expects;
    dinov2_root is the precompute_dinov2_native_pages.py output dir
    (contains a cls/ subdir keyed by <piece_id>_page_N.npy)."""
    import functools
    import os
    from extensions.hooks import mert_patch as _mp
    _mp._PATH_TO_EMB_ROOT = dict(path_to_emb_root)
    os.environ['MERT_PATH_MAP'] = ';'.join(f'{k}={v}' for k, v in path_to_emb_root.items())

    from audio_conditioned_unet import dataset as cpjku_dataset
    from audio_conditioned_unet import audio_encoder as cpjku_audio_encoder
    from extensions.audio_encoders.mert_dinov2_projector import MERTDINOv2Projector

    def _load_piece_with_dinov2_root(params):
        params = dict(params)
        params['dinov2_root'] = dinov2_root
        return _patched_load_piece_mert_dinov2(params)

    cpjku_dataset.load_piece = _load_piece_with_dinov2_root
    cpjku_audio_encoder.MERTDINOv2Projector = MERTDINOv2Projector
    print(f'[mert_dinov2_patch] Patched load_piece (MERT audio + DINOv2 CLS, '
          f'concatenated to 1536-dim) + registered MERTDINOv2Projector '
          f'(dinov2_root={dinov2_root})', flush=True)
