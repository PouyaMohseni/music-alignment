"""MERT+DINOv2 cross-attention entry point: MERT audio encoder + DINOv2
visual-token neck + TokenCrossAttentionFiLM as the fusion mechanism (audio
query attends over raw DINOv2 patch tokens, replacing FiLM), then runs
CPJKU's own train_model.py in-process via runpy.

Requires:
  DINOV2_TILED_ROOT -- output dir of scripts/precompute_dinov2_tiled_native.py
  MERT_PATH_MAP -- semicolon-separated dataset_path=emb_root pairs, same as
                   run_train_with_mert.py / run_train_b1a_cross_attention.py
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_dinov2_cross_attention_patch import patch_mert_dinov2_cross_attention
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

dinov2_root = os.environ.get('DINOV2_TILED_ROOT')
if not dinov2_root:
    raise RuntimeError('DINOV2_TILED_ROOT env var must be set, e.g. /scratch/pmohseni/dinov2_emb_tiled_native')

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
mert_path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

patch_mert_dinov2_cross_attention(dinov2_root=dinov2_root, mert_path_to_emb_root=mert_path_to_emb_root)
patch_lenient_load_state_dict()

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
