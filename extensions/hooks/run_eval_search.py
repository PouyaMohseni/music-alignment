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
              topk=int(os.environ.get('C2_TOPK', '32')),
              mu_pow=float(os.environ.get('TIME_MU_POW', '0')),
              sig_pow=float(os.environ.get('TIME_SIG_POW', '0')),
              ref_frames=float(os.environ.get('TIME_REF', '5')))
if KIND == 'beam':
    common['beam'] = int(os.environ.get('BEAM', '8'))
    common['cluster_px'] = float(os.environ.get('CLUSTER_PX', '0'))
elif KIND == 'viterbi':
    common['bin_px'] = float(os.environ.get('VIT_BIN', '8.0'))
    common['band_px'] = float(os.environ.get('VIT_BAND', '400.0'))

# probes must be installed BEFORE the search patch, so Detect/FiLM are already
# wrapped by the time anything reads their outputs
from extensions.hooks.cyolo_probe_patch import (configure, patch_int_scale_width,
                                                 patch_probes)

_drop = [s for s in os.environ.get('DROP_SCALES', '').split(',') if s != '']
configure(drop_scales=_drop,
          film_scale=float(os.environ.get('FILM_SCALE', '1.0')),
          sys_constrain=float(os.environ.get('SYS_SLACK', '0')))
if _drop or os.environ.get('FILM_SCALE', '1.0') != '1.0':
    patch_probes()
patch_int_scale_width()
_zmask = os.environ.get('Z_MASK', 'none')
if _zmask != 'none':
    from extensions.hooks.cyolo_probe_patch import patch_z_mask, set_z_mask
    set_z_mask(_zmask)
    patch_z_mask()

from extensions.hooks.cyolo_search_patch import patch_cyolo_search

patch_cyolo_search(kind=KIND, **common)

import cyolo_score_following.dataset as _d
import cyolo_score_following.utils.general as _g
if not getattr(_g, '_search_patched', False) or not getattr(_d, '_search_iterate_patched', False):
    raise RuntimeError('search patch did not take')

# architecture probes: recurrence timing, then the candidate-ceiling recorder.
# the oracle probe must go LAST so its get_max_box wrapper sees the same raw
# candidate tensor the decoder is handed.
_anchor = os.environ.get('ANCHOR', 'start')
_window = int(os.environ.get('WINDOW', '0'))
if _anchor != 'start' or _window:
    from extensions.hooks import cyolo_recur_patch as _rec
    _rec.configure(anchor=_anchor, window=_window)
    _rec.patch_encode_samples()
    from cyolo_score_following.models.conditioning_networks import ContextConditioning
    if not getattr(ContextConditioning, '_recur_patched', False):
        raise RuntimeError('recur patch did not take')

if os.environ.get('ORACLE', '0') == '1':
    from extensions.hooks import cyolo_oracle_probe as _orc
    _orc.patch_oracle()
    if not getattr(_d, '_oracle_patched', False):
        raise RuntimeError('oracle patch did not take')
    atexit.register(lambda: _orc.dump(os.environ['ORACLE_OUT']))

atexit.register(lambda: dump(os.environ['REC_OUT']))

_EVAL = os.path.join(_CY, 'cyolo_score_following', 'eval.py')
sys.argv[0] = _EVAL
os.chdir(os.path.dirname(_EVAL))
runpy.run_path(_EVAL, run_name='__main__')
