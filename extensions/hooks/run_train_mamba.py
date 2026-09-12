"""Train CYOLO with CODA's Mamba in place of the LSTM, or with the LSTM as the
budget-matched control.

    MAMBA_ENC=1 python -m extensions.hooks.run_train_mamba <train.py args>
    MAMBA_ENC=0 ...   the control, identical in every other respect

The control matters more than usual here. Our detector trainings reach about
ten epochs in a 24 hour A100 job, where Henkel's release is fully converged, so
a Mamba run compared against that release would confound the recurrence with
three orders of magnitude of compute. Only the two runs launched from this file
are compared with each other.
"""
import os
import runpy
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)

_CY = os.environ.get('CYOLO_ROOT', '/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0, _CY)

from extensions.hooks.numpy_compat import patch as _np_patch     # noqa: E402

_np_patch()

from extensions.hooks.mamba_patch import maybe_patch_mamba       # noqa: E402

_on = maybe_patch_mamba()

from cyolo_score_following.models.yolo import Model as _Model     # noqa: E402

if _on and not getattr(_Model, '_mamba_patched', False):
    raise RuntimeError('MAMBA_ENC is set but the patch did not take')
if not _on and getattr(_Model, '_mamba_patched', False):
    raise RuntimeError('MAMBA_ENC is off but the model is patched anyway')
print(f'[MAMBA] encoder arm: {"mamba" if _on else "lstm (control)"}', flush=True)

_TRAIN = os.path.join(_CY, 'cyolo_score_following', 'train.py')
sys.argv[0] = _TRAIN
os.chdir(os.path.dirname(_TRAIN))
runpy.run_path(_TRAIN, run_name='__main__')
