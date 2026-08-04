"""Zero-retrain eval wrapper: existing checkpoint + test-time adaptive input
normalisation (extensions/hooks/adaptive_norm_patch.py).

Nothing is trained and no weights change -- only the normalisation constants
applied to the audio tower's input. Run the SAME checkpoint with
ADAPTNORM_ALPHA=0 and ADAPTNORM_ALPHA=1 and the difference is attributable to
that one operator and nothing else.

MERT checkpoints additionally need patch_mert_pipeline, applied first so that
MERTProjector is registered on the audio_encoder module before the norm patch
looks for it. Detected from MERT_TEST_EMB_ROOT, exactly as eval_any_cpu.sh
sets it.
"""
import os
import runpy
import sys

REPO_ROOT = '/project/def-ichiro/pmohseni/music-alignment'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extensions.hooks.adaptive_norm_patch import patch_adaptive_input_norm
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

# A MERT checkpoint cannot be evaluated without its precomputed embeddings, so
# ADAPTNORM_IS_MERT is set explicitly by the caller rather than inferred from a
# possibly-stale MERT_TEST_EMB_ROOT -- pointing a real-audio run at the
# synthetic embedding root does not error, it silently scores synthetic audio.
if os.environ.get('ADAPTNORM_IS_MERT', '0') == '1':
    from extensions.hooks.mert_patch import patch_mert_pipeline

    emb_root = os.environ.get('MERT_TEST_EMB_ROOT')
    if not emb_root:
        raise RuntimeError('ADAPTNORM_IS_MERT=1 but MERT_TEST_EMB_ROOT is unset')
    test_dir = os.environ.get('MERT_EVAL_TEST_DIR')
    if not test_dir:
        raise RuntimeError('ADAPTNORM_IS_MERT=1 but MERT_EVAL_TEST_DIR is unset; '
                           'it must equal the --test_dir string verbatim')
    patch_mert_pipeline(path_to_emb_root={test_dir: emb_root})

patch_adaptive_input_norm()
patch_lenient_load_state_dict()

runpy.run_path(
    os.path.join(REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py'),
    run_name='__main__')
