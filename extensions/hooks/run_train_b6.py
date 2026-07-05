"""B6 entry point: patches midi_to_spec_otf with synthetic impulse-response
augmentation, then runs CPJKU's own unmodified train_model.py in-process
via runpy. No auxiliary loss (unlike B2-B5) -- this only changes what audio
the network sees during training, not the loss.

    python extensions/hooks/run_train_b6.py \
        --film_layers 2 3 4 5 6 7 8 --use_lstm --augment \
        --train_set ... --val_set ... --config ... \
        --audio_encoder CBEncoder --tag B6_impulse_response_aug
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.ir_patch import patch_ir_pipeline

AUG_PROB = float(os.environ.get('B6_AUG_PROB', '0.5'))
SNR_LO = float(os.environ.get('B6_SNR_LO', '10.0'))
SNR_HI = float(os.environ.get('B6_SNR_HI', '30.0'))
N_IRS = int(os.environ.get('B6_N_IRS', '16'))

patch_ir_pipeline(p=AUG_PROB, snr_range_db=(SNR_LO, SNR_HI), n_irs=N_IRS)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
