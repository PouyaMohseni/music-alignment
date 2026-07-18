"""MERT+DINOv2 (bottleneck-injection) entry point: frozen MERT audio +
a constant-per-page DINOv2 CLS embedding, concatenated into a single
1536-dim per-frame input (extensions/hooks/mert_dinov2_patch.py). The
existing from-scratch visual encoder/decoder/skip-connections are
completely untouched -- this only adds pretrained GLOBAL visual context
alongside the audio signal, the deliberately low-risk half of the visual-
pretraining experiment.

Requires MERT_PATH_MAP (same convention as run_train_with_mert.py) AND
DINOV2_ROOT (output dir of scripts/precompute_dinov2_native_pages.py).
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_dinov2_patch import patch_mert_dinov2_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

dinov2_root = os.environ.get('DINOV2_ROOT')
if not dinov2_root:
    raise RuntimeError('DINOV2_ROOT env var must be set, e.g. /scratch/pmohseni/dinov2_emb_native')

patch_mert_dinov2_pipeline(path_to_emb_root=path_to_emb_root, dinov2_root=dinov2_root)
patch_lenient_load_state_dict()

print(f'[run_train_mert_dinov2] Patched load_piece (MERT+DINOv2 concat) '
      f'(dinov2_root={dinov2_root})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
