"""V-DINOv2 entry point: replaces ConditionalUNet's entire from-scratch
visual encoder with a DINOv2-tile-grid neck (extensions/hooks/
dinov2_full_encoder_patch.py), keeping plain CBEncoder audio (not MERT) to
isolate this as a pure visual-architecture change. Then runs CPJKU's own
train_model.py in-process via runpy.

Requires DINOV2_TILED_ROOT env var (output dir of
scripts/precompute_dinov2_tiled_native.py).
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.dinov2_full_encoder_patch import patch_dinov2_full_encoder
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

dinov2_root = os.environ.get('DINOV2_TILED_ROOT')
if not dinov2_root:
    raise RuntimeError('DINOV2_TILED_ROOT env var must be set, e.g. /scratch/pmohseni/dinov2_emb_tiled_native')

patch_dinov2_full_encoder(dinov2_root=dinov2_root)
patch_lenient_load_state_dict()

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
