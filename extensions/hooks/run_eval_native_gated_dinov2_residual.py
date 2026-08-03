"""Eval wrapper for Gated-residual DINOv2 hybrid (extensions/hooks/dinov2_full_encoder_patch.py):
same runpy-into-eval_model.py pattern as run_eval_native_mert.py. Plain
CBEncoder audio (no MERT) -- only DINOV2_TILED_ROOT is needed, and it's a
single flat root keyed by piece_name.npy covering train+test alike, so no
test-specific subdir/env var is required (unlike MERT_TEST_EMB_ROOT).
"""
import os
import runpy
import sys

REPO_ROOT = '/project/def-ichiro/pmohseni/music-alignment'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_CPJKU_ROOT = os.path.join(REPO_ROOT, 'third_party', 'cpjku_unet')
if _CPJKU_ROOT not in sys.path:
    sys.path.insert(0, _CPJKU_ROOT)

from extensions.hooks.gated_dinov2_residual_patch import patch_gated_dinov2_residual
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

DINOV2_TILED_ROOT = os.environ.get('DINOV2_TILED_ROOT')
if not DINOV2_TILED_ROOT:
    raise RuntimeError('DINOV2_TILED_ROOT env var must be set, e.g. '
                        '/scratch/pmohseni/dinov2_emb_tiled_native')

patch_gated_dinov2_residual(dinov2_root=DINOV2_TILED_ROOT)
patch_lenient_load_state_dict()

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
