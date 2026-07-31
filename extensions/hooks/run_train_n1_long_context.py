"""N1 entry point -- B1a's frozen-MERT audio encoder + a two-tier memory
Transformer REPLACING CB_TA's 1-layer LSTM temporal core
(extensions/heads/long_context_temporal.py). Visual encoder and FiLM are
untouched.

Warm start: everything except the temporal core is restored from B1a's
converged checkpoint, so only the new temporal module trains from scratch.
`rnn.` is whitelisted as missing because the LSTM's parameters
(rnn.weight_ih_l0, ...) are genuinely gone -- replaced by the memory
Transformer's own. Every other base key must still be present, which
warm_start_load enforces.

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
from extensions.hooks.temporal_arch_patch import patch_long_context_temporal
from extensions.hooks.warm_start_load import patch_warm_start_load

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

patch_mert_pipeline(path_to_emb_root=path_to_emb_root)
patch_long_context_temporal(
    n_layers=int(os.environ.get('N1_LAYERS', '2')),
    n_heads=int(os.environ.get('N1_HEADS', '8')),
    n_fine=int(os.environ.get('N1_N_FINE', '64')),
    n_comp=int(os.environ.get('N1_N_COMP', '192')),
    pool=int(os.environ.get('N1_POOL', '16')),
)
patch_warm_start_load(allow_missing_prefixes=('rnn.',))

runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
