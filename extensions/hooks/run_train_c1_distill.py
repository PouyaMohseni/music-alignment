"""C1 entry point -- posteriorgram distillation into cyolo_sb's audio encoder.

Fine-tunes the RELEASED cyolo_sb checkpoint (the 79.9 model, verified in our own
harness) with one added training-time loss: the conditioning vector z must
predict the AMT posteriorgram at its own frame. Inference is untouched -- same
78-band mel encoder, same detector, same decode, head deleted.

Rationale in extensions/heads/posteriorgram_distill.py. Short version: the AMT
representation loses 0.001 onset F1 to the room where our trackers lose ~30
points, and every attempt to USE that representation as an input added capacity
and overfit on 353 pieces. Distillation transfers the invariance at zero
inference capacity.
"""
import os, runpy, sys

_T = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_T, '..', '..'))
_CY = os.environ.get('CYOLO_ROOT', '/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0, _R)
sys.path.insert(0, _CY)

from extensions.hooks.cyolo_aux_patch import patch_cyolo_aux


def _parse_map(s):
    return dict(p.split('=', 1) for p in s.split(';') if p)


bm = os.environ.get('C1_BANK_MAP')
if not bm:
    raise RuntimeError('C1_BANK_MAP must be set: "dataset_dir=post_dir;..."')

patch_cyolo_aux(_parse_map(bm), weight=float(os.environ.get('C1_WEIGHT', '1.0')))

# Verify by SENTINEL, not by function name: a name check passes whether or not
# the patch applied (that mistake shipped once already).
import cyolo_score_following.dataset as _ds
import cyolo_score_following.utils.loss as _loss
if not getattr(_ds.SequenceDataset, '_c1_patched', False) or \
        not getattr(_loss, '_c1_patched', False) or \
        not getattr(_ds, '_c1_iterate_patched', False):
    raise RuntimeError('C1 patch did not take')
print('[C1] verified: dataset, loss and iterate_dataset are patched', flush=True)

_TRAIN = os.path.join(_CY, 'cyolo_score_following', 'train.py')
sys.argv[0] = _TRAIN
os.chdir(os.path.dirname(_TRAIN))
runpy.run_path(_TRAIN, run_name='__main__')
