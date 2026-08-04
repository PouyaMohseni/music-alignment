"""R2 entry point -- multi-condition MERT training: half the performances are
served from the CLEAN embedding bank, half from the ACOUSTICALLY DEGRADED one
(scripts/precompute_mert_augmented.py: random spectral tilt + room IR + noise,
re-encoded through MERT).

WHY THIS IS THE HIGHEST-CEILING TRACK. Our models lose ~45 points going from
synthetic MSMD to a room microphone (B1a: 90.0 -> 38.5) while CYOLO loses 4.3.
That is a domain-shift failure, and multi-condition training is the standard
fix for domain shift. B6 already tried an approximation of this and came LAST
on `room` (15.6), but it was crippled two ways: it augmented the CBEncoder
branch, which trails MERT by ~20 points on room before any augmentation, and
it applied reverb only, when the dominant synth->real difference is a static
per-band gain. R2 fixes both.

WHY THE MODEL SEES DEGRADED AUDIO IT CANNOT "UNDO". The augmented bank was
encoded by pushing degraded WAVEFORMS through frozen MERT, so the degradation
is baked into the features the network consumes. It cannot be inverted by a
normalisation layer -- which is precisely why this is a stronger intervention
than R1 (test-time CMN, which turned out to be worth at most +1.5).

WARM START. From B1a_mert_swap (the clean MERT base, 38.5 on room) and NOT
from the stronger MERT_B2_pitch_aux, so the delta this run produces is
attributable to multi-condition training alone rather than confounded with the
pitch auxiliary loss. No parameters are added or removed -- only which .npy a
performance is read from -- so the checkpoint stays a plain B1a-shaped model
and `run_eval_native_mert.py` evaluates it with no special handling.

Requires MERT_PATH_MAP (clean) and MERT_AUG_PATH_MAP + MERT_AUG_PROB
(degraded). All three are validated below rather than trusted: a silently
inactive augmentation would produce a run that looks like R2, is reported as
R2, and is actually just B1a fine-tuned for longer.
"""
import os
import runpy
import sys
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')
sys.path.insert(0, _PROJECT_ROOT)

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                       '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)

# ---------------------------------------------------------------------------
# Provenance guard. The failure this prevents is silent, not loud: if
# MERT_AUG_PATH_MAP is unset/misspelled or the bank is incomplete,
# _load_mert_spec quietly serves clean embeddings for everything and the run is
# indistinguishable from plain B1a in every log line and every metric -- except
# it would be written up as multi-condition training.
# ---------------------------------------------------------------------------
aug_map_str = os.environ.get('MERT_AUG_PATH_MAP', '')
aug_map = dict(pair.split('=', 1) for pair in aug_map_str.split(';') if pair)
aug_prob = float(os.environ.get('MERT_AUG_PROB', '0'))

if not aug_map:
    raise RuntimeError('MERT_AUG_PATH_MAP is empty -- this would train on CLEAN '
                       'embeddings only while claiming to be multi-condition R2')
if aug_prob <= 0.0:
    raise RuntimeError(f'MERT_AUG_PROB={aug_prob} -- the degraded bank would never '
                       f'be used; set it to e.g. 0.5')

TRAIN_SET = os.environ.get('R2_TRAIN_SET')
if TRAIN_SET and TRAIN_SET not in aug_map:
    raise RuntimeError(f'train set {TRAIN_SET!r} has no entry in MERT_AUG_PATH_MAP '
                       f'({list(aug_map)}) -- augmentation would silently not apply '
                       f'to the training data, only possibly to validation')

for ds_path, aug_root in aug_map.items():
    n_aug = len(list(Path(aug_root).glob('*.npy'))) if Path(aug_root).is_dir() else 0
    clean_root = path_to_emb_root.get(ds_path)
    n_clean = len(list(Path(clean_root).glob('*.npy'))) if clean_root and Path(clean_root).is_dir() else 0
    print(f'[R2] {ds_path}\n     clean={clean_root} ({n_clean} npy)'
          f'\n     aug  ={aug_root} ({n_aug} npy)', flush=True)
    if n_aug == 0:
        raise RuntimeError(f'augmented bank {aug_root!r} holds no .npy files')
    # A partial bank is allowed (the loader falls back to clean per key and warns)
    # but must be surfaced here, because it directly dilutes the intervention.
    if n_clean and n_aug < 0.98 * n_clean:
        print(f'[R2] WARNING: augmented bank covers only {100.0 * n_aug / n_clean:.1f}% '
              f'of the clean bank; the effective augmentation rate is below '
              f'MERT_AUG_PROB={aug_prob}', flush=True)

print(f'[R2] multi-condition training ACTIVE: p(degraded)={aug_prob}, '
      f'assigned deterministically per (piece, tempo)', flush=True)

patch_mert_pipeline(path_to_emb_root=path_to_emb_root)
patch_lenient_load_state_dict()

runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
