"""R2a entry point -- B1a's frozen-MERT pipeline plus time-constant channel
augmentation in embedding space (extensions/hooks/channel_aug_patch.py).

Warm-starts from B1a_mert_swap, NOT from the stronger MERT_B2_pitch_aux, on
purpose: B1a is the clean MERT base at 38.5 pct@0.5s on `room`, so the delta
this run produces is attributable to the augmentation alone. Warm-starting
from the pitch-aux model would score higher and mean nothing, because the gain
could not be separated from the pitch loss.

No parameters are added, so the checkpoint loads with plain lenient_load and
eval needs no special wrapper -- run_eval_native_mert.py covers it (the patch
is gated on self.training and is a no-op under eval()).

Requires MERT_PATH_MAP (same convention as run_train_with_mert.py).
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.channel_aug_patch import patch_mert_channel_aug
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                       '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

# MERT pipeline first: it registers MERTProjector on the audio_encoder module,
# which the augmentation patch then wraps.
patch_mert_pipeline(path_to_emb_root=path_to_emb_root)
patch_mert_channel_aug()
patch_lenient_load_state_dict()

runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
