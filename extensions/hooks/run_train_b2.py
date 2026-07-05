"""B2 entry point: patches the data pipeline to carry per-timestep active
MIDI pitches (pitch_patch.py) AND patches iterate_dataset with the pitch
auxiliary loss (decoder stage 6, matching CB_TA-Ext.md's spec exactly --
unlike B5, this one maps cleanly under network.py's numbering), then runs
CPJKU's own unmodified train_model.py in-process via runpy.

    python extensions/hooks/run_train_b2.py \
        --film_layers 2 3 4 5 6 7 8 --use_lstm --augment \
        --train_set ... --val_set ... --config ... \
        --audio_encoder CBEncoder --tag B2_pitch_aux
"""
import functools
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.pitch_patch import patch_pitch_pipeline
patch_pitch_pipeline()

from extensions.hooks.iterate_dataset_ext import iterate_dataset_ext
from extensions.losses.b2_callback import b2_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

AUX_LOSS_WEIGHT = float(os.environ.get('B2_AUX_LOSS_WEIGHT', '0.3'))
DECODER_STAGE = int(os.environ.get('B2_DECODER_STAGE', '6'))

cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=b2_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT,
    need_rnn_capture=True, decoder_feature_stage=DECODER_STAGE, need_pitch_roll=True)

print(f'[run_train_b2] Patched load_piece/ScoreAudioDataset (pitch roll) + iterate_dataset '
      f'(weight={AUX_LOSS_WEIGHT}, decoder_stage={DECODER_STAGE})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
