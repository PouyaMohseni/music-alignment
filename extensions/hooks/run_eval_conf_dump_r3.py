"""R3 (MERT + gated belief propagation) eval that additionally dumps per-frame
confidence signals and tracking error, for the calibration study.

Identical to run_eval_native_n3_belief_propagation.py -- same patches, same
eval_model.py, so the printed pct@0.5s must reproduce
results/eval_any-111639.log -- plus confidence_dump_patch, which only *observes*
the segmentation heatmap.  Output NPZ path comes from CONF_DUMP_OUT.
"""
import os
import runpy
import sys

REPO_ROOT = '/project/def-ichiro/pmohseni/music-alignment'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_CPJKU_ROOT = os.path.join(REPO_ROOT, 'third_party', 'cpjku_unet')
if _CPJKU_ROOT not in sys.path:
    sys.path.insert(0, _CPJKU_ROOT)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.temporal_arch_patch import patch_gated_belief_propagation
from extensions.hooks.lenient_load import patch_lenient_load_state_dict
from extensions.hooks.confidence_dump_patch import patch_confidence_dump

MERT_TEST_EMB_ROOT = os.environ.get('MERT_TEST_EMB_ROOT')
if not MERT_TEST_EMB_ROOT:
    raise RuntimeError('MERT_TEST_EMB_ROOT env var must be set')

TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')
OUT = os.environ.get('CONF_DUMP_OUT')
if not OUT:
    raise RuntimeError('CONF_DUMP_OUT env var must be set')

patch_mert_pipeline(path_to_emb_root={TEST_DIR: MERT_TEST_EMB_ROOT})
patch_gated_belief_propagation(
    belief_h=int(os.environ.get('N3_BELIEF_H', '16')),
    belief_w=int(os.environ.get('N3_BELIEF_W', '64')),
)
patch_lenient_load_state_dict()
patch_confidence_dump(OUT)

runpy.run_path(os.path.join(_CPJKU_ROOT, 'audio_conditioned_unet', 'eval_model.py'),
               run_name='__main__')
