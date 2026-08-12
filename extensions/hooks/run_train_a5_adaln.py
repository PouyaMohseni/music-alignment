"""A5 entry point -- adaLN-Zero in place of FiLM, everything else identical.

An ABLATION, not a bet: a seven-way controlled comparison found FiLM, AdaLN,
adaLN-Zero, cross-attention, prefix, adaRMSNorm and additive injection all
comparable under normal training. It is worth one run because adaLN-Zero is the
variant that consistently beats cross-attention (which we measured losing badly
here: 19.3 and 2.6 on room), it is FiLM-shaped, and its zero-init starts as an
exact no-op -- the property our collapsed cross-attention runs lacked.
"""
import os, runpy, sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
_CPJKU = os.path.join(_ROOT, 'third_party', 'cpjku_unet')
sys.path.insert(0, _ROOT)
sys.path.insert(0, _CPJKU)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict
from extensions.heads.adaln_zero import patch_adaln_zero

pm = os.environ.get('MERT_PATH_MAP')
if not pm:
    raise RuntimeError('MERT_PATH_MAP must be set')
path_to_emb_root = dict(p.split('=', 1) for p in pm.split(';') if p)

# FiLM must be swapped BEFORE the network is constructed, or the modules are
# already built from the original class.
patch_adaln_zero(norm=os.environ.get('A5_NORM', 'group'))
patch_mert_pipeline(path_to_emb_root=path_to_emb_root)
patch_lenient_load_state_dict()

from audio_conditioned_unet import network as _net
from extensions.heads.adaln_zero import AdaLNZero
if not issubclass(_net.FiLM, AdaLNZero):
    raise RuntimeError('A5 patch did NOT take: network.FiLM is not AdaLNZero')
print('[A5] verified: network.FiLM is AdaLNZero', flush=True)

runpy.run_path(os.path.join(_CPJKU, 'audio_conditioned_unet', 'train_model.py'),
               run_name='__main__')
