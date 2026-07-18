"""MERT+noise entry point: frozen MERT audio (B1a's load_piece patch,
unchanged) + SNR-controlled Gaussian noise injected on the embedding vector
during training (extensions/audio_encoders/mert_projector_noisy.py),
loosely approximating B6's impulse-response augmentation spirit without a
true acoustic-domain reproduction -- see that module's docstring for why.

Purely a network-level change (registers MERTProjectorNoisy as a new
audio_encoder option), so load_piece is untouched -- no composition risk
with anything else.

Requires MERT_PATH_MAP env var (same convention as run_train_with_mert.py).
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

os.environ['MERT_PATH_MAP'] = path_map_str
from extensions.hooks import mert_patch as _mp
_mp._PATH_TO_EMB_ROOT = dict(path_to_emb_root)

from audio_conditioned_unet import dataset as cpjku_dataset
from audio_conditioned_unet import audio_encoder as cpjku_audio_encoder
from extensions.audio_encoders.mert_projector_noisy import MERTProjectorNoisy

cpjku_dataset.load_piece = _mp._patched_load_piece
cpjku_audio_encoder.MERTProjectorNoisy = MERTProjectorNoisy

from extensions.hooks.lenient_load import patch_lenient_load_state_dict
patch_lenient_load_state_dict()

print(f'[run_train_mert_noisy] Patched load_piece (MERT audio, unchanged) + '
      f'registered MERTProjectorNoisy (path_to_emb_root={path_to_emb_root})', flush=True)

_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
