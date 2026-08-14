"""Run the cyolo eval with per-frame recording, C2 optionally on.

Same harness, same metric, same weights in both arms -- the ONLY difference is
whether the C2 decode patch is installed (C2_ON=1). That is what makes the
resulting per-piece arrays a paired sample.
"""
import atexit
import os
import runpy
import sys

_R = '/project/def-ichiro/pmohseni/music-alignment'
_CY = os.environ.get('CYOLO_ROOT', '/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0, _R)
sys.path.insert(0, _CY)

from extensions.hooks.cyolo_record_perframe import patch_recorder, dump

patch_recorder()

_c2 = os.environ.get('C2_ON', '0') == '1'
if _c2:
    from extensions.hooks.cyolo_temporal_patch import patch_cyolo_temporal
    patch_cyolo_temporal(
        lam=float(os.environ.get('C2_LAM', '1.0')),
        fwd_px=float(os.environ.get('C2_FWD', '6.0')),
        sigma_px=float(os.environ.get('C2_SIGMA', '18.0')),
        jump_logp=float(os.environ.get('C2_JUMP', '-6.0')),
    )
    import cyolo_score_following.dataset as _d
    import cyolo_score_following.utils.general as _g
    if not getattr(_g, '_c2_patched', False) or not getattr(_d, '_c2_iterate_patched', False):
        raise RuntimeError('C2 patch did not take')
print(f'[REC] arm = {"C2" if _c2 else "BASELINE"}', flush=True)

# C2 wraps iterate_dataset and the recorder wraps the two stat functions it
# calls, so ordering matters: the recorder must be installed FIRST (above) so
# C2's wrapper sits outside it and both arms record identically.
_OUT = os.environ['REC_OUT']
atexit.register(lambda: dump(_OUT))

_EVAL = os.path.join(_CY, 'cyolo_score_following', 'eval.py')
sys.argv[0] = _EVAL
os.chdir(os.path.dirname(_EVAL))
runpy.run_path(_EVAL, run_name='__main__')
