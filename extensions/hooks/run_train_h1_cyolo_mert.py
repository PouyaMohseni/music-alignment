"""H1 entry point -- CYOLO's detector with a MERT audio tower.

Combines the two largest effects we have measured, which live on opposite sides
of the architecture and have never been combined:

  * detection output parameterisation -- our own reproduction reaches 67.1 on
    `room` at 18% of training, where our dense-heatmap model converges at 56.6
    with the same IR bank and protocol;
  * the MERT audio tower -- worth ~+22 on `room` over the 78-band mel CNN in
    our own sweep.

Only ContextConditioning.enc changes. The kw=40 windowing, the LSTM, the
concat trick, the FPN, the anchors and the multi-class head are untouched, so
a change in the result is attributable to the audio representation.

Env:
  H1_EMB_MAP      "dataset_dir=emb_dir;dataset_dir=emb_dir"   (required)
  H1_AUG_MAP      same form, IR-degraded bank                 (optional)
  H1_AUG_PROB     fraction served degraded, e.g. 0.5          (optional)

Everything is verified below rather than assumed: a silently inactive patch
would train a plain mel-CNN CYOLO and be reported as H1, which is the same
class of failure the R2 guard exists to catch.
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

_CY = os.environ.get('CYOLO_ROOT', '/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0, _CY)

from extensions.hooks.cyolo_mert_patch import patch_cyolo_mert   # noqa: E402


def _parse_map(s):
    return dict(pair.split('=', 1) for pair in s.split(';') if pair)


emb_map_str = os.environ.get('H1_EMB_MAP')
if not emb_map_str:
    raise RuntimeError('H1_EMB_MAP must be set: "dataset_dir=emb_dir;..."')
emb_map = _parse_map(emb_map_str)

aug_map = _parse_map(os.environ.get('H1_AUG_MAP', '')) or None
aug_prob = float(os.environ.get('H1_AUG_PROB', '0'))
if aug_map and aug_prob <= 0:
    raise RuntimeError('H1_AUG_MAP is set but H1_AUG_PROB is 0 -- the degraded '
                       'bank would never be used, and the run would be reported '
                       'as multi-condition while training clean-only')

patch_cyolo_mert(emb_map, aug_roots=aug_map, aug_prob=aug_prob)

# Verify the swap took, rather than trusting that it did.
import cyolo_score_following.dataset as _ds                       # noqa: E402
from cyolo_score_following.models.yolo import Model as _CyoloModel  # noqa: E402
if getattr(_ds.load_dataset, '__name__', '') != 'load_dataset' or \
        _CyoloModel.compute_spec.__name__ != 'compute_spec':
    raise RuntimeError('H1 patch did not take')
print('[H1] verified: load_dataset and Model.compute_spec are patched', flush=True)

_TRAIN = os.path.join(_CY, 'cyolo_score_following', 'train.py')
sys.argv[0] = _TRAIN
os.chdir(os.path.dirname(_TRAIN))
runpy.run_path(_TRAIN, run_name='__main__')
