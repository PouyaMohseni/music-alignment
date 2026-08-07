"""Eval wrapper for MERT+DINOv2 TokenCrossAttentionFiLM (extensions/hooks/
mert_dinov2_cross_attention_patch.py): same runpy-into-eval_model.py pattern
as run_eval_native_mert.py, applying both the MERT audio swap and the DINOv2
visual + cross-attention fusion patch together.
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

from extensions.hooks.mert_dinov2_cross_attention_patch import patch_mert_dinov2_cross_attention
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

DINOV2_TILED_ROOT = os.environ.get('DINOV2_TILED_ROOT')
if not DINOV2_TILED_ROOT:
    raise RuntimeError('DINOV2_TILED_ROOT env var must be set, e.g. '
                        '/scratch/pmohseni/dinov2_emb_tiled_native')

MERT_TEST_EMB_ROOT = os.environ.get('MERT_TEST_EMB_ROOT')
if not MERT_TEST_EMB_ROOT:
    raise RuntimeError('MERT_TEST_EMB_ROOT env var must be set, e.g. '
                        '/scratch/pmohseni/mert_emb_zenodo/msmd_test')

TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')

patch_mert_dinov2_cross_attention(
    dinov2_root=DINOV2_TILED_ROOT,
    mert_path_to_emb_root={TEST_DIR: MERT_TEST_EMB_ROOT})
patch_lenient_load_state_dict()

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
