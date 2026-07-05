"""B3 entry point: patches iterate_dataset with the INR sub-pixel refinement
aux loss at decoder stage 9 (= self.decoder[0], the layer just before
conv_out -- CB_TA-Ext.md's "decoder_final", and unlike B5 this one DOES map
cleanly under network.py's combined FiLM-stage numbering), then runs
CPJKU's own unmodified train_model.py in-process via runpy.

CB_TA-Ext.md notes this ablation "can fine-tune on top of a converged
base" -- pass --param_path pointing at A0's best_model.pt to warm-start
rather than training the coarse decoder from scratch, since the refiner's
"coarse peak" input is meaningless noise until the base heatmap is
reasonably localized.

    python extensions/hooks/run_train_b3.py \
        --film_layers 2 3 4 5 6 7 8 --use_lstm --augment \
        --train_set ... --val_set ... --config ... \
        --audio_encoder CBEncoder --tag B3_inr_subpixel [--param_path A0_best.pt]
"""
import functools
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.iterate_dataset_ext import iterate_dataset_ext
from extensions.losses.b3_callback import b3_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

AUX_LOSS_WEIGHT = float(os.environ.get('B3_AUX_LOSS_WEIGHT', '1.0'))
DECODER_STAGE = int(os.environ.get('B3_DECODER_STAGE', '9'))   # decoder_final

cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=b3_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT,
    decoder_feature_stage=DECODER_STAGE)

print(f'[run_train_b3] Patched iterate_dataset with INR sub-pixel refinement '
      f'(weight={AUX_LOSS_WEIGHT}, decoder_stage={DECODER_STAGE})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
