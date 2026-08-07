"""B1a + gated cross-attention FiLM entry point: B1a's frozen-MERT audio-
encoder swap (extensions/hooks/mert_patch.py) PLUS replacing FiLM with
GatedSpatialCrossAttentionFiLM (extensions/hooks/
gated_cross_attention_film_patch.py) -- isolates whether B1a-cross-
attention's underperformance (71.1% pct@0.5s vs B1a-gated-film's 82.9%) was
due to the cross-attention mechanism itself or due to lacking gated-film's
zero-init stabilization. Visual side is still CB_TA's own from-scratch conv
encoder, unchanged -- only the conditioning mechanism changes.

Requires MERT_PATH_MAP env var, same as run_train_with_mert.py.
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')

sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.gated_cross_attention_film_patch import patch_gated_cross_attention_film

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)
patch_mert_pipeline(path_to_emb_root=path_to_emb_root)
patch_gated_cross_attention_film()

runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
