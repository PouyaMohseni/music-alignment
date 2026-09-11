"""Serve the MERT-audio detector (H1) through the evaluation harnesses.

Set H1_EMB_MAP="dataset_dir=emb_dir;..." and pass H1's checkpoint as
--param_path; nothing else changes. The detector's audio input becomes the
precomputed MERT bank for each piece exactly as in training
(run_train_h1_cyolo_mert.py), with no multi-condition draw and no feature
augmentation: the bank named in the map is the one served.
"""
from __future__ import annotations

import os


def maybe_patch_h1() -> bool:
    s = os.environ.get('H1_EMB_MAP', '')
    if not s:
        return False
    if os.environ.get('IR_PATH'):
        # cyolo convolves WAVEFORMS with the IR; an embedding cannot be
        # convolved, so the degradation has to be baked into the bank instead
        raise RuntimeError('H1_EMB_MAP and IR_PATH are exclusive: use a degraded bank')
    from extensions.hooks.cyolo_mert_patch import patch_cyolo_mert
    emb = dict(pair.split('=', 1) for pair in s.split(';') if pair)
    patch_cyolo_mert(emb)
    import cyolo_score_following.dataset as ds
    from cyolo_score_following.models.yolo import Model
    if not getattr(ds, '_h1_patched', False) or not getattr(Model, '_h1_patched', False):
        raise RuntimeError('H1 patch did not take')
    print(f'[H1] eval serving MERT banks: {emb}', flush=True)
    return True
