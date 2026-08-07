"""Eval wrapper for B1a + SpatialCrossAttentionFiLM (extensions/hooks/
cross_attention_film_patch.py): same runpy-into-eval_model.py pattern as
run_eval_native_mert.py, plus the cross-attention FiLM replacement patch on
top of the frozen-MERT audio swap.
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

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.cross_attention_film_patch import patch_cross_attention_film
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

MERT_TEST_EMB_ROOT = os.environ.get('MERT_TEST_EMB_ROOT')
if not MERT_TEST_EMB_ROOT:
    raise RuntimeError('MERT_TEST_EMB_ROOT env var must be set, e.g. '
                        '/scratch/pmohseni/mert_emb_zenodo/msmd_test')

TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')

patch_mert_pipeline(path_to_emb_root={TEST_DIR: MERT_TEST_EMB_ROOT})
patch_cross_attention_film()
patch_lenient_load_state_dict()

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
