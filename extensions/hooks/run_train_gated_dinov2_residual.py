"""Gated-residual DINOv2 hybrid entry point: keeps CB_TA's own from-scratch
visual encoder as the primary path, adds DINOv2 tile-grid features only as
a zero-initialized additive residual (extensions/hooks/
gated_dinov2_residual_patch.py) -- tests whether V-DINOv2's catastrophic
full-replacement failure (6.9% pct@0.5s) was a resolution-loss problem
specific to discarding the original encoder, or whether DINOv2 signal is
useless here regardless of how it's integrated. Plain CBEncoder audio (NOT
MERT), matching V-DINOv2's isolation discipline.

Requires DINOV2_TILED_ROOT env var (output dir of
scripts/precompute_dinov2_tiled_native.py).
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.gated_dinov2_residual_patch import patch_gated_dinov2_residual
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

dinov2_root = os.environ.get('DINOV2_TILED_ROOT')
if not dinov2_root:
    raise RuntimeError('DINOV2_TILED_ROOT env var must be set, e.g. /scratch/pmohseni/dinov2_emb_tiled_native')

patch_gated_dinov2_residual(dinov2_root=dinov2_root)
patch_lenient_load_state_dict()

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
