"""MERT+DINOv2-aware native eval: applies patch_mert_dinov2_pipeline (same
patch used for training -- MERT audio + constant-per-page DINOv2 CLS,
concatenated to 1536-dim) before running eval_model.py via runpy.

Requires MERT_TEST_EMB_ROOT and DINOV2_ROOT env vars.
"""
import os
import runpy
import sys

REPO_ROOT = '/project/def-ichiro/pmohseni/music-alignment'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extensions.hooks.mert_dinov2_patch import patch_mert_dinov2_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

MERT_TEST_EMB_ROOT = os.environ.get('MERT_TEST_EMB_ROOT')
if not MERT_TEST_EMB_ROOT:
    raise RuntimeError('MERT_TEST_EMB_ROOT env var must be set')
DINOV2_ROOT = os.environ.get('DINOV2_ROOT')
if not DINOV2_ROOT:
    raise RuntimeError('DINOV2_ROOT env var must be set')

TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')

patch_mert_dinov2_pipeline(path_to_emb_root={TEST_DIR: MERT_TEST_EMB_ROOT}, dinov2_root=DINOV2_ROOT)
patch_lenient_load_state_dict()

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
