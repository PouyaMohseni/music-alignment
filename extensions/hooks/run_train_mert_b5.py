"""MERT+B5 entry point: combines the frozen MERT audio-encoder swap (B1a,
mert_patch.py -- patches load_piece + registers MERTProjector) with B5's
dense-contrastive auxiliary loss (patches iterate_dataset). Orthogonal
patches, same reasoning as run_train_mert_c2.py -- see that file's docstring
for why B2/B6 specifically are NOT safely combinable this way.

Tests whether B5's dense-contrastive training objective (84.8% pct@0.5s
with the original CBEncoder) helps when combined with MERT features
instead of CBEncoder's.

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
from extensions.losses.b5_callback import b5_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)
patch_mert_pipeline(path_to_emb_root=path_to_emb_root)

patch_lenient_load_state_dict()

AUX_LOSS_WEIGHT = float(os.environ.get('B5_AUX_LOSS_WEIGHT', '0.2'))
DECODER_STAGE = int(os.environ.get('B5_DECODER_STAGE', '7'))
cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=b5_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT,
    need_rnn_capture=True, decoder_feature_stage=DECODER_STAGE)

print(f'[run_train_mert_b5] Patched load_piece+MERTProjector (MERT audio) '
      f'+ iterate_dataset (dense_contrastive_aux_loss, weight={AUX_LOSS_WEIGHT}, '
      f'decoder_stage={DECODER_STAGE})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
