"""B5 entry point: patches iterate_dataset with the dense-contrastive aux
loss (sampled from decoder stage 5, per CB_TA-Ext.md's B5 config), then runs
CPJKU's own unmodified train_model.py in-process via runpy.

    python extensions/hooks/run_train_b5.py \
        --film_layers 2 3 4 5 6 7 8 --use_lstm --augment \
        --train_set ... --val_set ... --config ... \
        --audio_encoder CBEncoder --tag B5_dense_contrastive_aux
"""
import functools
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.lenient_load import patch_lenient_load_state_dict
patch_lenient_load_state_dict()

from extensions.hooks.iterate_dataset_ext import iterate_dataset_ext
from extensions.losses.b5_callback import b5_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

# CB_TA-Ext.md specifies feature_map_stage: decoder_5, but under network.py's
# actual combined FiLM-stage numbering (film_layers spans encoder+bottleneck+
# decoder as one 1..2*(n_encoder_layers+1) sequence), stage 5 IS the
# bottleneck for n_encoder_layers=4 -- there is no decoder stage 5. Valid
# decoder stages for the default 4-encoder-layer config are 6-9 (self.decoder
# indices 3,2,1,0 respectively). Using 7 here (one stage earlier than B2's
# decoder_6) as the nearest reasonable substitute, distinct from B2's stage
# so the two auxiliary losses aren't reading the identical feature map.
AUX_LOSS_WEIGHT = float(os.environ.get('B5_AUX_LOSS_WEIGHT', '0.2'))
DECODER_STAGE = int(os.environ.get('B5_DECODER_STAGE', '7'))

cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=b5_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT,
    need_rnn_capture=True, decoder_feature_stage=DECODER_STAGE)

print(f'[run_train_b5] Patched iterate_dataset with dense_contrastive_aux_loss '
      f'(weight={AUX_LOSS_WEIGHT}, decoder_stage={DECODER_STAGE})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
