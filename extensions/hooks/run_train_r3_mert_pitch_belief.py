"""R3 -- stack the two ingredients that independently give the largest
real-audio gains, which have never been combined.

From the 2026-08-03 sweep (REAL_AUDIO_SWEEP.md, pct@0.5s on `room`), both are
measured against the same B1a/MERT base (38.5):

    N3_belief_propagation   44.7   (+6.2)   gated Bayes filter on the output
    MERT_B2_pitch_aux       43.7   (+5.2)   pitch-roll auxiliary loss
    B1a_mert_swap           38.5     --     base

They attack different things and should be close to additive:

  * The belief filter is a TEMPORAL prior. It constrains where the position
    can move between frames and keeps an explicit escape probability, so a
    momentary bad audio frame cannot teleport the estimate across the page.
    Degraded audio produces exactly that kind of momentary bad frame, which
    is why it is the best real-audio model we have.
  * The pitch auxiliary loss is a REPRESENTATIONAL constraint on the audio
    tower. Forcing the FiLM features to remain predictive of the pitch roll
    stops the tower from leaning on channel/timbre cues that do not survive a
    change of piano and room.

Neither touches the other's parameters, so this is a genuine stack rather than
two things competing for the same capacity.

WARM START. The belief filter's gate is zero-initialised and the LSTM keeps
stock nn.LSTM parameter names, so loading MERT_B2_pitch_aux's converged
checkpoint reproduces that model exactly at step zero; only `belief_filter.`
is whitelisted as missing. Every other base key must still be present, so a
silently-mismatched checkpoint still fails loudly.
"""
import functools
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_pitch_patch import patch_mert_pitch_pipeline
from extensions.hooks.temporal_arch_patch import patch_gated_belief_propagation
from extensions.hooks.warm_start_load import patch_warm_start_load
from extensions.hooks.iterate_dataset_ext import iterate_dataset_ext
from extensions.losses.b2_callback import b2_aux_loss

import audio_conditioned_unet.dataset as cpjku_dataset

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                       '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

# Order matters: the MERT+pitch pipeline registers MERTProjector and the
# pitch_roll field on the piece dict, and the belief filter patch rebuilds
# ConditionalUNet -- so the data pipeline must be in place first.
patch_mert_pitch_pipeline(path_to_emb_root=path_to_emb_root)
patch_gated_belief_propagation(
    belief_h=int(os.environ.get('N3_BELIEF_H', '16')),
    belief_w=int(os.environ.get('N3_BELIEF_W', '64')),
)
patch_warm_start_load(allow_missing_prefixes=('belief_filter.',))

AUX_LOSS_WEIGHT = float(os.environ.get('B2_AUX_LOSS_WEIGHT', '0.3'))
DECODER_STAGE = int(os.environ.get('B2_DECODER_STAGE', '6'))
cpjku_dataset.iterate_dataset = functools.partial(
    iterate_dataset_ext, aux_loss_fn=b2_aux_loss, aux_loss_weight=AUX_LOSS_WEIGHT,
    need_rnn_capture=True, decoder_feature_stage=DECODER_STAGE, need_pitch_roll=True)

print(f'[run_train_r3] MERT+pitch pipeline + gated belief filter + pitch aux '
      f'(weight={AUX_LOSS_WEIGHT}, decoder_stage={DECODER_STAGE})', flush=True)

runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
