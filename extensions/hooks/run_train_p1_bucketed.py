"""P1 entry point -- bucketed-softmax position objective on the MERT base.

Replaces the dense soft-Dice heatmap loss with a softmax over x columns
(extensions/heads/bucketed_softmax.py explains why this is the highest-value
remaining experiment).  The network body, the audio tower and the FiLM stack
are untouched; only the objective and the decode change.

Warm-starts from R2r_realir, our best real-audio model (56.6 room), so the
delta this run produces is attributable to the output parameterisation on top
of everything already won -- not confounded with IR augmentation, which
R2r_realir already contains.

MERT_PATH_MAP is required, exactly as for the other MERT runs.  The bucketed
patch is verified ACTIVE below rather than assumed: a silently inactive patch
would fall through to plain Dice and produce a run that looks like P1, is
reported as P1, and is really just R2r fine-tuned for longer -- the same class
of silent failure the R2 guard was written to catch.
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
_CPJKU_REPO = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet')
_CPJKU_PKG_DIR = os.path.join(_CPJKU_REPO, 'audio_conditioned_unet')
sys.path.insert(0, _PROJECT_ROOT)
# Make `audio_conditioned_unet` importable as a package EXPLICITLY. The other
# entry points get this implicitly and it is not obvious how, so relying on it
# cost job 551057 an A100 allocation. Being explicit is idempotent and removes
# the dependency on whatever that mechanism is.
sys.path.insert(0, _CPJKU_REPO)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict
from extensions.hooks.bucketed_softmax_patch import patch_bucketed_softmax

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                       '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

POOL = os.environ.get('P1_POOL', 'logsumexp')
DICE_WEIGHT = float(os.environ.get('P1_DICE_WEIGHT', '0'))

if POOL not in ('logsumexp', 'mean', 'max'):
    raise RuntimeError(f'P1_POOL={POOL!r} is not one of logsumexp/mean/max')

patch_mert_pipeline(path_to_emb_root=path_to_emb_root)
patch_lenient_load_state_dict()
patch_bucketed_softmax(pool=POOL, dice_weight=DICE_WEIGHT)

# Verify the swap actually took, rather than trusting that it did.
import audio_conditioned_unet.dataset as _ds
_fn = getattr(_ds, 'iterate_dataset', None)
_name = getattr(getattr(_fn, 'func', _fn), '__name__', '?')
if _name != 'iterate_dataset_bucketed':
    raise RuntimeError(f'bucketed patch did NOT take: dataset.iterate_dataset is {_name!r}. '
                       f'This run would silently train plain soft-Dice and be reported as P1.')
print(f'[P1] verified: dataset.iterate_dataset -> {_name} '
      f'(pool={POOL}, dice_weight={DICE_WEIGHT})', flush=True)

runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
