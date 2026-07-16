"""Wrapper for CPJKU's native eval_model.py that applies the lenient_load
patch before running it, so checkpoints with extension-only aux modules
(B2's _ext_b2_pitch_head, B3's _ext_b3_inr_refiner, B5's _ext_b5_audio_proj
-- see extensions/hooks/lenient_load.py) don't crash strict state_dict
loading. eval_model.py's own load call has no strict= kwarg to override,
and it's a __main__-guarded script with no importable entrypoint, so this
execs its body directly (same technique already used for train_model.py by
the other extensions/hooks/run_train_*.py wrappers) after patching.

Run exactly like eval_model.py itself, just via this wrapper:
    python /path/to/extensions/hooks/run_eval_native.py --param_path ... --test_dir ... [...]
"""
import os
import sys

REPO_ROOT = '/project/def-ichiro/pmohseni/music-alignment'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extensions.hooks.lenient_load import patch_lenient_load_state_dict
patch_lenient_load_state_dict()

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

with open(_EVAL_MODEL_PATH) as f:
    _code = f.read()

sys.argv[0] = _EVAL_MODEL_PATH
exec(compile(_code, _EVAL_MODEL_PATH, 'exec'), {'__name__': '__main__'})
