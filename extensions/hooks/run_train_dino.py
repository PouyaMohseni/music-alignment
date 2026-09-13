"""Train CYOLO with DINOv2 as its visual stem, or the matched control.

    DINO_ENC=1 DINO_NOSHIFT=1 python -m extensions.hooks.run_train_dino <args>
    DINO_ENC=0 DINO_NOSHIFT=1 ...   the control, identical otherwise

Both arms drop the page-shift augmentation, because a precomputed patch map can
only be rolled in whole 8-px steps and the note anchors are 11 px wide. The
control therefore cannot be the LSTM run launched earlier, which had shifts on.
"""
import os
import runpy
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', '..')))
_CY = os.environ.get('CYOLO_ROOT', '/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0, _CY)

from extensions.hooks.numpy_compat import patch as _np_patch     # noqa: E402

_np_patch()

from extensions.hooks.dino_backbone_patch import maybe_patch_dino  # noqa: E402

_on = maybe_patch_dino()

import cyolo_score_following.dataset as _ds                        # noqa: E402
from cyolo_score_following.models.yolo import Model as _Model      # noqa: E402

if _on and not (getattr(_ds, '_dino_patched', False)
                and getattr(_Model, '_dino_patched', False)):
    raise RuntimeError('DINO_ENC is set but the patch did not take')
if os.environ.get('DINO_NOSHIFT', '0') != '0' and not getattr(_ds, '_noshift_patched', False):
    raise RuntimeError('DINO_NOSHIFT is set but the shift patch did not take')
print(f'[DINO] visual arm: {"dinov2" if _on else "cnn (control)"}', flush=True)

_TRAIN = os.path.join(_CY, 'cyolo_score_following', 'train.py')
sys.argv[0] = _TRAIN
os.chdir(os.path.dirname(_TRAIN))
runpy.run_path(_TRAIN, run_name='__main__')
