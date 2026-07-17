"""MERT-aware native eval: applies patch_mert_pipeline (same monkey-patch
B1a/MERT+B5/MERT+C2 use for training) before running CPJKU's own unmodified
eval_model.py via runpy, so MERTProjector-audio_encoder checkpoints get
evaluated against the correct per-native-page MERT embeddings instead of
eval_model.py's default live spectrogram computation (which the network
was never trained to consume for these checkpoints).

Uses runpy.run_path(..., run_name='__main__'), NOT exec() -- see
extensions/hooks/run_eval_native.py's docstring for why exec() causes a
multiprocessing respawn storm with eval_model.py's mp.set_start_method('spawn').

Requires MERT_TEST_EMB_ROOT env var (e.g. /scratch/pmohseni/mert_emb_zenodo/msmd_test).
Run exactly like eval_model.py itself, just via this wrapper:
    MERT_TEST_EMB_ROOT=/scratch/pmohseni/mert_emb_zenodo/msmd_test \
    python /path/to/extensions/hooks/run_eval_native_mert.py \
        --param_path ... --test_dir ../data/msmd/msmd_test [...]
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

# Must match the exact --test_dir string eval_model.py's argparse receives
# (mert_patch._get_path_to_emb_root looks up by that literal string).
TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')

patch_mert_pipeline(path_to_emb_root={TEST_DIR: MERT_TEST_EMB_ROOT})
patch_lenient_load_state_dict()

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
