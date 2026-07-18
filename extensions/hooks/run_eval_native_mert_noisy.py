"""Same as run_eval_native_mert.py, but also registers MERTProjectorNoisy --
MERT_noisy's checkpoint uses audio_encoder=MERTProjectorNoisy (extensions/
audio_encoders/mert_projector_noisy.py), which the plain MERT-aware eval
wrapper doesn't know about (it only registers MERTProjector). Note the
noise injection only fires in .train() mode (self.training), so eval here
sees the clean embedding exactly like plain MERTProjector would -- this
tests whether training WITH noise produced a more robust model, not
whether noise helps at eval time.
"""
import os
import runpy
import sys

REPO_ROOT = '/project/def-ichiro/pmohseni/music-alignment'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

MERT_TEST_EMB_ROOT = os.environ.get('MERT_TEST_EMB_ROOT')
if not MERT_TEST_EMB_ROOT:
    raise RuntimeError('MERT_TEST_EMB_ROOT env var must be set, e.g. '
                        '/scratch/pmohseni/mert_emb_zenodo/msmd_test')

TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')

patch_mert_pipeline(path_to_emb_root={TEST_DIR: MERT_TEST_EMB_ROOT})
patch_lenient_load_state_dict()

from audio_conditioned_unet import audio_encoder as cpjku_audio_encoder
from extensions.audio_encoders.mert_projector_noisy import MERTProjectorNoisy
cpjku_audio_encoder.MERTProjectorNoisy = MERTProjectorNoisy
print('[run_eval_native_mert_noisy] Registered MERTProjectorNoisy', flush=True)

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
