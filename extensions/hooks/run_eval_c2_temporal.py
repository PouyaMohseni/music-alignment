"""C2 eval -- released cyolo_sb, decoded with the causal temporal filter.

No training. Same weights, same features, same metric; only the read-out of a
position from the per-frame detections changes. Any delta is attributable to
the decision rule alone.
"""
import os, runpy, sys
_R = '/project/def-ichiro/pmohseni/music-alignment'
_CY = os.environ.get('CYOLO_ROOT', '/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0, _R); sys.path.insert(0, _CY)

from extensions.hooks.cyolo_temporal_patch import patch_cyolo_temporal
patch_cyolo_temporal(
    lam=float(os.environ.get('C2_LAM', '1.0')),
    fwd_px=float(os.environ.get('C2_FWD', '6.0')),
    sigma_px=float(os.environ.get('C2_SIGMA', '18.0')),
    jump_logp=float(os.environ.get('C2_JUMP', '-6.0')),
)
import cyolo_score_following.utils.general as _g
import cyolo_score_following.dataset as _d
if not getattr(_g, '_c2_patched', False) or not getattr(_d, '_c2_iterate_patched', False):
    raise RuntimeError('C2 patch did not take')
print('[C2] verified: get_max_box and iterate_dataset are patched', flush=True)

_EVAL = os.path.join(_CY, 'cyolo_score_following', 'eval.py')
sys.argv[0] = _EVAL
os.chdir(os.path.dirname(_EVAL))
runpy.run_path(_EVAL, run_name='__main__')
