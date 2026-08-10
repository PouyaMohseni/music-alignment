"""Throwaway probe: does the P1 patch chain work under the real sbatch
invocation (cwd inside the cpjku package, entry point by absolute path)?
Kept out of the training path; safe to delete."""
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
_CPJKU_REPO = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet')
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CPJKU_REPO)
print('cwd:', os.getcwd())
from extensions.hooks.bucketed_softmax_patch import patch_bucketed_softmax
print('OK import')
patch_bucketed_softmax(pool='logsumexp', dice_weight=0.0)
import audio_conditioned_unet.dataset as ds
fn = getattr(ds, 'iterate_dataset')
name = getattr(getattr(fn, 'func', fn), '__name__', '?')
assert name == 'iterate_dataset_bucketed', name
print('PATCH VERIFIED ->', name)
# also exercise the eval-side decode patch
import torch
from audio_conditioned_unet.network import ConditionalUNet
from extensions.heads.bucketed_softmax import decode_mask
d = decode_mask(torch.randn(2, 1, 8, 40), 8)
print('decode ok', tuple(d.shape), 'max=%.3f' % float(d.max()))
print('ALL GOOD')
