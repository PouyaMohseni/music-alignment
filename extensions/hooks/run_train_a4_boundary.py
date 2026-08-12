"""A4+A3 entry point -- boundary output + coarse staff head on the MERT base.

Warm-starts from R2r_realir (56.6 room, our best), so the delta measures the
OUTPUT reformulation on top of everything already won -- MERT tower and real-IR
augmentation -- rather than re-deriving those gains.

Env: MERT_PATH_MAP (required); A4_STAGE, A4_BINS, A4_W_STAFF, A4_DICE optional.
The patch is verified ACTIVE below: a silently inactive one would train plain
soft-Dice and be reported as A4.
"""
import os, runpy, sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
_CPJKU = os.path.join(_ROOT, 'third_party', 'cpjku_unet')
sys.path.insert(0, _ROOT)
sys.path.insert(0, _CPJKU)          # makes audio_conditioned_unet importable

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict
from extensions.hooks.boundary_patch import patch_boundary

pm = os.environ.get('MERT_PATH_MAP')
if not pm:
    raise RuntimeError('MERT_PATH_MAP must be set')
path_to_emb_root = dict(p.split('=', 1) for p in pm.split(';') if p)

patch_mert_pipeline(path_to_emb_root=path_to_emb_root)
patch_lenient_load_state_dict()
patch_boundary(decoder_stage=int(os.environ.get('A4_STAGE', '6')),
               n_staff_bins=int(os.environ.get('A4_BINS', '16')),
               w_staff=float(os.environ.get('A4_W_STAFF', '1.0')),
               dice_weight=float(os.environ.get('A4_DICE', '0')))

import audio_conditioned_unet.dataset as _ds
_fn = getattr(_ds, 'iterate_dataset', None)
_name = getattr(getattr(_fn, 'func', _fn), '__name__', '?')
if _name != 'iterate_dataset_boundary':
    raise RuntimeError(f'A4 patch did NOT take: iterate_dataset is {_name!r}')
print(f'[A4] verified: iterate_dataset -> {_name}', flush=True)

runpy.run_path(os.path.join(_CPJKU, 'audio_conditioned_unet', 'train_model.py'),
               run_name='__main__')
