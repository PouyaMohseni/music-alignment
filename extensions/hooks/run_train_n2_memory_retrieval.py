"""N2 entry point -- B1a's frozen-MERT audio encoder + a zero-init-gated,
lag-aware retrieval read over the piece's own compressed history
(extensions/heads/gated_memory_retrieval.py). The LSTM, visual encoder and
FiLM are all untouched.

Warm start: because the LSTM keeps stock nn.LSTM parameter names and the
retrieval gate is zero-initialised, loading B1a's converged checkpoint makes
this network compute EXACTLY B1a at step zero (asserted in
scripts/smoke_test_temporal_arch.py). Only `mem_read.` is whitelisted as
missing; every other base key must still be present.

Requires MERT_PATH_MAP (same convention as run_train_with_mert.py).
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.temporal_arch_patch import patch_gated_memory_retrieval
from extensions.hooks.warm_start_load import patch_warm_start_load

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

patch_mert_pipeline(path_to_emb_root=path_to_emb_root)
patch_gated_memory_retrieval(
    n_heads=int(os.environ.get('N2_HEADS', '8')),
    n_mem=int(os.environ.get('N2_N_MEM', '192')),
    pool=int(os.environ.get('N2_POOL', '16')),
)
patch_warm_start_load(allow_missing_prefixes=('mem_read.',))

runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
