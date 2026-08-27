"""Run cyolo_sb over a split and dump its candidate boxes for training a selector.

No decoder is installed: we want the raw per-frame candidate set, not a decoded
trajectory. The batch frame-index patch IS installed, because elapsed time
between consecutive scored frames is one of the features (under --only_onsets a
step is one onset to the next, and those are 1 to 64 frames apart).
"""
import atexit
import os
import runpy
import sys

_R = '/project/def-ichiro/pmohseni/music-alignment'
_CY = os.environ.get('CYOLO_ROOT', '/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0, _R)
sys.path.insert(0, _CY)

from extensions.hooks.numpy_compat import patch as _np_patch

_np_patch()          # cyolo's loader uses np.float / np.int

from extensions.hooks.cyolo_probe_patch import patch_int_scale_width
from extensions.hooks.cyolo_search_patch import patch_batch_frames

patch_int_scale_width()

import cyolo_score_following.dataset as _d

patch_batch_frames()

_ir = os.environ.get('IR_PATH', '')
if _ir:
    from extensions.hooks.piece_ir_patch import patch_loader_ir, patch_piece_ir
    patch_piece_ir(seed=int(os.environ.get('IR_SEED', '0')),
                   prob=float(os.environ.get('IR_PROB', '1.0')))
    patch_loader_ir(_ir.split(','))
    if not getattr(_d, '_ir_loader_patched', False):
        raise RuntimeError('IR loader patch did not take')

from extensions.hooks import cyolo_cand_dump as _cd

_cd.patch_dump()
if not getattr(_d, '_dump_patched', False):
    raise RuntimeError('dump patch did not take')
if not getattr(_d.CustomBatch, '_frames_patched', False):
    raise RuntimeError('frame-index patch did not take')

atexit.register(lambda: _cd.dump(os.environ['DUMP_OUT']))

_EVAL = os.path.join(_CY, 'cyolo_score_following', 'eval.py')
sys.argv[0] = _EVAL
os.chdir(os.path.dirname(_EVAL))
runpy.run_path(_EVAL, run_name='__main__')
