"""Eval wrapper for N1 (two-tier memory Transformer temporal core) -- applies the same patches as
run_train_n1_long_context.py, then runs CPJKU's own unmodified eval_model.py
via runpy (same pattern as run_eval_native_mert.py).

The new temporal module's weights ARE in the checkpoint being evaluated, so
no key whitelist is needed here; lenient_load is used only to tolerate
extension-only keys, exactly as the other native eval wrappers do.
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
from extensions.hooks.temporal_arch_patch import patch_long_context_temporal
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

MERT_TEST_EMB_ROOT = os.environ.get('MERT_TEST_EMB_ROOT')
if not MERT_TEST_EMB_ROOT:
    raise RuntimeError('MERT_TEST_EMB_ROOT env var must be set, e.g. '
                        '/scratch/pmohseni/mert_emb_zenodo/msmd_test')

TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')

patch_mert_pipeline(path_to_emb_root={TEST_DIR: MERT_TEST_EMB_ROOT})
patch_long_context_temporal(
    n_layers=int(os.environ.get('N1_LAYERS', '2')),
    n_heads=int(os.environ.get('N1_HEADS', '8')),
    n_fine=int(os.environ.get('N1_N_FINE', '64')),
    n_comp=int(os.environ.get('N1_N_COMP', '192')),
    pool=int(os.environ.get('N1_POOL', '16')),
)
patch_lenient_load_state_dict()

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
