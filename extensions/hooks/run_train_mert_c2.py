"""MERT+C2 entry point: combines the frozen MERT audio-encoder swap (B1a,
mert_patch.py -- patches load_piece + registers MERTProjector) with C2's
soft-DTW auxiliary loss (patches iterate_dataset). These two patches are
orthogonal (different module attributes: load_piece vs iterate_dataset),
confirmed by reading both extensions/hooks/mert_patch.py and
extensions/hooks/run_train_c2.py before combining -- unlike B2 (which ALSO
patches load_piece via pitch_patch.py for pitch-roll data and would
silently conflict with MERT's own load_piece patch) or B6 (which patches
spectrogram computation directly, which doesn't exist once MERT bypasses
live spectrogram synthesis entirely).

Tests whether C2's soft-DTW training objective (85.0% pct@0.5s with the
original CBEncoder) helps when combined with MERT features instead, since
frozen MERT alone (B1a) has historically underperformed CBEncoder on this
task -- the auxiliary loss might still transfer even if raw MERT features
don't beat CBEncoder on their own.

Requires MERT_PATH_MAP env var (same convention as run_train_with_mert.py).
"""
import functools
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict
from extensions.hooks.iterate_dataset_ext import iterate_dataset_ext
from extensions.losses.c2_callback import c2_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)
patch_mert_pipeline(path_to_emb_root=path_to_emb_root)

patch_lenient_load_state_dict()

AUX_LOSS_WEIGHT = float(os.environ.get('C2_AUX_LOSS_WEIGHT', '1.0'))
cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=c2_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT)

print(f'[run_train_mert_c2] Patched load_piece+MERTProjector (MERT audio) '
      f'+ iterate_dataset (soft_dtw_loss, weight={AUX_LOSS_WEIGHT})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
