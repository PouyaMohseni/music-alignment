"""Eval cyolo_sb with a search decoder, recording per frame for paired analysis."""
import atexit
import os
import runpy
import sys

_R = '/project/def-ichiro/pmohseni/music-alignment'
_CY = os.environ.get('CYOLO_ROOT', '/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0, _R)
sys.path.insert(0, _CY)

from extensions.hooks.cyolo_record_perframe import dump, patch_recorder

patch_recorder()

KIND = os.environ.get('SEARCH_KIND', 'beam')
common = dict(lam=float(os.environ.get('C2_LAM', '1.0')),
              fwd_px=float(os.environ.get('C2_FWD', '6.0')),
              sigma_px=float(os.environ.get('C2_SIGMA', '18.0')),
              jump_logp=float(os.environ.get('C2_JUMP', '-6.0')),
              topk=int(os.environ.get('C2_TOPK', '32')))
if KIND == 'beam':
    common['beam'] = int(os.environ.get('BEAM', '8'))
elif KIND == 'viterbi':
    common['bin_px'] = float(os.environ.get('VIT_BIN', '8.0'))
    common['band_px'] = float(os.environ.get('VIT_BAND', '400.0'))

from extensions.hooks.cyolo_search_patch import patch_cyolo_search

patch_cyolo_search(kind=KIND, **common)

import cyolo_score_following.dataset as _d
import cyolo_score_following.utils.general as _g
if not getattr(_g, '_search_patched', False) or not getattr(_d, '_search_iterate_patched', False):
    raise RuntimeError('search patch did not take')

atexit.register(lambda: dump(os.environ['REC_OUT']))

_EVAL = os.path.join(_CY, 'cyolo_score_following', 'eval.py')
sys.argv[0] = _EVAL
os.chdir(os.path.dirname(_EVAL))
runpy.run_path(_EVAL, run_name='__main__')
