"""B4 entry point: patches iterate_dataset with the temporal-consistency aux
loss, then runs CPJKU's own unmodified train_model.py in-process via runpy.
Base network/CBEncoder/dice loss are all untouched -- only an extra loss
term is added on top, exactly as CB_TA-Ext.md specifies ("added on top of
Dice, not instead of it").

    python extensions/hooks/run_train_b4.py \
        --film_layers 2 3 4 5 6 7 8 --use_lstm --augment \
        --train_set ... --val_set ... --config ... \
        --audio_encoder CBEncoder --tag B4_temporal_consistency
"""
import functools
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.iterate_dataset_ext import iterate_dataset_ext
from extensions.losses.b4_callback import b4_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

AUX_LOSS_WEIGHT = float(os.environ.get('B4_AUX_LOSS_WEIGHT', '1.0'))

cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=b4_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT)

print(f'[run_train_b4] Patched iterate_dataset with temporal_consistency_loss '
      f'(weight={AUX_LOSS_WEIGHT})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
