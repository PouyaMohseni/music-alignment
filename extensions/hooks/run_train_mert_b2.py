"""MERT+B2 entry point: combines frozen MERT audio (instead of a live
spectrogram) with pitch-roll data AND B2's pitch-auxiliary loss.

Unlike MERT+B3/B5/C2 (which only needed to compose two orthogonal patches),
B2 ALSO patches load_piece (via pitch_patch.py, to carry per-timestep MIDI
pitch data alongside spec) -- naively applying both B1a's and B2's
load_piece patches would mean whichever ran last silently wins, with no
error to catch it. extensions/hooks/mert_pitch_patch.py properly merges
them into one load_piece that produces both a MERT-embedding 'spec' and a
'pitch_roll' from the same MIDI file. See that module's docstring.

Requires MERT_PATH_MAP env var (same convention as run_train_with_mert.py).
"""
import functools
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_pitch_patch import patch_mert_pitch_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict
from extensions.hooks.iterate_dataset_ext import iterate_dataset_ext
from extensions.losses.b2_callback import b2_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)
patch_mert_pitch_pipeline(path_to_emb_root=path_to_emb_root)

patch_lenient_load_state_dict()

AUX_LOSS_WEIGHT = float(os.environ.get('B2_AUX_LOSS_WEIGHT', '0.3'))
DECODER_STAGE = int(os.environ.get('B2_DECODER_STAGE', '6'))
cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=b2_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT,
    need_rnn_capture=True, decoder_feature_stage=DECODER_STAGE, need_pitch_roll=True)

print(f'[run_train_mert_b2] Patched load_piece (MERT audio + pitch_roll) + '
      f'iterate_dataset (weight={AUX_LOSS_WEIGHT}, decoder_stage={DECODER_STAGE})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
