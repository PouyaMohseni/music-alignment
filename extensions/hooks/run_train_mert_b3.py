"""MERT+B3 entry point: combines the frozen MERT audio-encoder swap (B1a,
mert_patch.py -- patches load_piece + registers MERTProjector) with B3's
INR sub-pixel refinement auxiliary loss (patches iterate_dataset). Same
orthogonality reasoning as run_train_mert_c2.py/run_train_mert_b5.py -- B3
only touches iterate_dataset, confirmed by reading extensions/hooks/run_train_b3.py
before combining.

B3's own docstring notes it "can fine-tune on top of a converged base"
since the refiner's coarse-peak input is meaningless noise until the base
heatmap is already reasonably localized -- this run is intended to be
warm-started from B1a's OWN converged checkpoint (88.9% pct@0.5s, the best
result in the project as of 2026-07-17), via --param_path, rather than
from scratch. The INR refiner module is created LAZILY on the first
training call (see extensions/hooks/iterate_dataset_ext.py's docstring),
i.e. after checkpoint loading and optimizer construction -- so loading
B1a's checkpoint (no _ext_b3_inr_refiner.* keys yet) into a fresh network
(which also has no refiner attribute yet, before the first forward call)
loads cleanly with no key mismatch at all.

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
from extensions.losses.b3_callback import b3_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)
patch_mert_pipeline(path_to_emb_root=path_to_emb_root)

patch_lenient_load_state_dict()

AUX_LOSS_WEIGHT = float(os.environ.get('B3_AUX_LOSS_WEIGHT', '0.1'))
DECODER_STAGE = int(os.environ.get('B3_DECODER_STAGE', '9'))
cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=b3_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT,
    decoder_feature_stage=DECODER_STAGE)

print(f'[run_train_mert_b3] Patched load_piece+MERTProjector (MERT audio) '
      f'+ iterate_dataset (INR sub-pixel refinement, weight={AUX_LOSS_WEIGHT}, '
      f'decoder_stage={DECODER_STAGE})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
