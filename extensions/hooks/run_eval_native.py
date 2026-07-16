"""Wrapper for CPJKU's native eval_model.py that applies the lenient_load
patch before running it, so checkpoints with extension-only aux modules
(B2's _ext_b2_pitch_head, B3's _ext_b3_inr_refiner, B5's _ext_b5_audio_proj
-- see extensions/hooks/lenient_load.py) don't crash strict state_dict
loading. eval_model.py's own load call has no strict= kwarg to override,
and it's a __main__-guarded script with no importable entrypoint.

IMPORTANT: this MUST use runpy.run_path (same technique as the working
extensions/hooks/run_train_*.py wrappers), NOT exec(compile(...)). A first
version used exec() directly and caused a catastrophic multiprocessing
respawn storm (jobs 65735183/84/85, 1.3M+ log lines / 21k+ respawns before
being killed): eval_model.py calls mp.set_start_method('spawn', force=True)
and uses a Pool for dataset loading; the 'spawn' start method needs to
reconstruct sys.modules['__main__'] in each worker, and exec()'ing code
into a fake {'__name__': '__main__'} globals dict leaves the REAL
sys.modules['__main__'].__spec__ in a broken state that the spawn
bootstrap can't recover from, so every worker crashed immediately and got
endlessly respawned. runpy.run_path sets up run_name='__main__' properly
enough for multiprocessing's bootstrap to work (confirmed by
run_train_b4.py etc. already doing exactly this against train_model.py,
which also spawns worker processes, for many hours without incident).

Run exactly like eval_model.py itself, just via this wrapper:
    python /path/to/extensions/hooks/run_eval_native.py --param_path ... --test_dir ... [...]
"""
import os
import runpy
import sys

REPO_ROOT = '/project/def-ichiro/pmohseni/music-alignment'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extensions.hooks.lenient_load import patch_lenient_load_state_dict
patch_lenient_load_state_dict()

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
