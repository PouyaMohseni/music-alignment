"""B1 -- monkey-patch CPJKU's data pipeline to use precomputed frozen MERT
embeddings instead of a live mel-spectrogram, and register MERTProjector as
a selectable audio_encoder. Base files (network.py, dataset.py, utils.py)
are never edited -- only monkey-patched via module attribute reassignment.

Why a full load_piece replacement rather than patching midi_to_spec_otf:
midi_to_spec_otf(midi, spec_params, sound_font_path) receives an in-memory
MIDIFile object, not the (piece, tempo_factor) identity needed to look up
which precomputed .npy to load -- that identity only exists as a local
variable inside load_performance/load_piece. And since load_dataset spawns
fresh worker processes via multiprocessing.Pool with start_method='spawn'
(train_model.py sets this explicitly), any patch applied only in the parent
process's sys.modules would not be visible in worker processes -- but a
patched module-level function *object* (e.g.
`audio_conditioned_unet.dataset.load_piece = patched_load_piece`) is looked
up fresh from the module's namespace by load_dataset's `pool.map(load_piece,
params)` at call time, and pickled by reference for spawn, so this works
correctly across the process boundary as long as patched_load_piece is
itself an ordinary, fully self-contained, importable function (it is).

Usage (before calling anything from audio_conditioned_unet):
    from extensions.hooks.mert_patch import patch_mert_pipeline
    patch_mert_pipeline(mert_emb_root='/scratch/pmohseni/mert_emb_zenodo/train_full')
"""
from __future__ import annotations
import copy
import os
from pathlib import Path

import numpy as np

# Maps each dataset path (the exact string passed as --train_set/--val_set)
# to its precomputed-MERT directory. train_model.py calls load_dataset twice
# (train_set, then val_set) in the SAME process, each needing a DIFFERENT
# embeddings root -- a single global root would silently use the wrong
# directory for one of the two.
#
# NOTE: this dict must NOT be relied on inside spawned worker processes.
# train_model.py uses multiprocessing.set_start_method('spawn'), so workers
# re-import mert_patch.py fresh from disk -- they never see patch_mert_pipeline()'s
# runtime mutation of this global (only reassigned module-level function
# *objects*, like load_piece below, are pickled by reference and survive the
# spawn boundary; plain runtime-set globals reset to their source-level value,
# i.e. empty, in the fresh import). So _load_mert_spec reads the MERT_PATH_MAP
# env var instead, which IS inherited by spawned children.
_PATH_TO_EMB_ROOT: dict[str, str] = {}


def _get_path_to_emb_root() -> dict[str, str]:
    path_map_str = os.environ.get('MERT_PATH_MAP', '')
    return dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)


def _load_mert_spec(dataset_path: str, piece: str, tempo_factor) -> np.ndarray:
    """Load precomputed MERT embedding, transposed to CPJKU's (768, T) spec
    convention (precompute_mert_zenodo.py saves (T, 768))."""
    path_to_emb_root = _get_path_to_emb_root()
    emb_root = path_to_emb_root.get(dataset_path)
    if emb_root is None:
        raise KeyError(f'No MERT embeddings root registered for dataset path {dataset_path!r} '
                        f'-- known paths: {list(path_to_emb_root)} '
                        f'(MERT_PATH_MAP={os.environ.get("MERT_PATH_MAP")!r})')
    key = f'{piece}_tempo_{tempo_factor}' if tempo_factor != -1 else piece
    path = Path(emb_root) / f'{key}.npy'
    emb = np.load(path).astype(np.float32)   # (T, 768)
    return emb.T                              # (768, T)


def _patched_load_performance(path, piece, spectrogram_params, coords, coord2onset, sf_path,
                               tempo_factor=1., real_perf=False, transpose=0):
    """Reimplements audio_conditioned_unet.utils.load_performance, substituting
    a precomputed MERT embedding for the live-synthesized mel-spectrogram.
    Onset/coordinate logic is copied verbatim -- only the `spec` source changes."""
    import os
    from madmom.io import midi as mm_midi
    from scipy import interpolate
    from audio_conditioned_unet.utils import merge_onsets

    if real_perf:
        midi_path = os.path.join(path, 'performance', piece + f'_{tempo_factor}.wav')  # unused for spec
        midi_path = os.path.join(path, 'performance', piece + '.mid')
    else:
        if tempo_factor == -1:
            midi_path = os.path.join(path, 'performance', piece + '.mid')
        else:
            midi_path = os.path.join(path, 'performance', piece + f'_tempo_{tempo_factor}.mid')

    midi = mm_midi.MIDIFile(midi_path)

    if transpose != 0:
        notes = midi.notes
        notes[:, 1] += transpose
        midi = mm_midi.MIDIFile.from_notes(notes)

    spec = _load_mert_spec(path, piece, tempo_factor)         # (768, T) -- the only change
    spec = np.pad(spec, ((0, 0), (spectrogram_params['pad'], 0)), mode='constant')

    onsets = (midi.notes[:, 0] * spectrogram_params['fps']).astype(int)

    onsets, coords_new = merge_onsets(onsets, copy.deepcopy(coords), coord2onset[0])
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return spec, onsets, coords_new, interpol_fnc


def _patched_load_piece(params):
    """Reimplements audio_conditioned_unet.dataset.load_piece, calling the
    unmodified load_score but the MERT-substituted load_performance above."""
    import numpy as np
    from audio_conditioned_unet.utils import load_score

    i = params['i']
    path = params['path']
    piece_name = params['piece_name']
    spectrogram_params = params['spectrogram_params']
    scale_factor = params.get('scale_factor', 3)
    tempo_factors = params['tempo_factors']
    transpose = params.get('transpose', 0)

    org_score_res, score, coords, coord2onset = load_score(path, piece_name, scale_factor)

    performances = {}

    for tempo_factor in tempo_factors:
        spec, onsets, coords_new, interpol_fnc = _patched_load_performance(
            path, piece_name, spectrogram_params, coords, coord2onset,
            sf_path=params['sf_path'], tempo_factor=tempo_factor,
            real_perf=params['real_perf'], transpose=transpose)

        unrolled_coords_x = []
        coords_per_staff = []
        max_xes = [0]
        staff_coords = sorted(np.unique(coords_new[:, 0]))

        for c in staff_coords:
            cs_staff = coords_new[coords_new[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        add_per_staff = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords)):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        from scipy import interpolate
        interpol_c2o = interpolate.interp1d(unrolled_coords_x, onsets, kind='previous', bounds_error=False,
                                            fill_value=(onsets[0], onsets[-1]))

        performances[tempo_factor] = {'interpol_fnc': interpol_fnc,
                                      'spec': spec,
                                      'onsets': onsets,
                                      'interpol_c2o': interpol_c2o,
                                      'add_per_staff': [staff_coords, add_per_staff]
                                      }

    return i, score, piece_name, performances


def patch_mert_pipeline(path_to_emb_root: dict[str, str]):
    """Call once, before load_dataset()/ConditionalUNet() -- registers
    MERTProjector as a selectable audio_encoder and replaces load_piece so
    the dataset pipeline reads precomputed MERT instead of live spectrograms.

    path_to_emb_root: maps each dataset path (the exact string passed as
    --train_set/--val_set/--test_dir) to its precomputed-MERT directory,
    e.g. {'/scratch/pmohseni/msmd_train_full': '/scratch/pmohseni/mert_emb_zenodo/train_full',
          '../data/msmd/msmd_valid': '/scratch/pmohseni/mert_emb_zenodo/msmd_valid'}
    """
    global _PATH_TO_EMB_ROOT
    _PATH_TO_EMB_ROOT = dict(path_to_emb_root)
    os.environ['MERT_PATH_MAP'] = ';'.join(f'{k}={v}' for k, v in _PATH_TO_EMB_ROOT.items())

    from audio_conditioned_unet import dataset as cpjku_dataset
    from audio_conditioned_unet import audio_encoder as cpjku_audio_encoder
    from extensions.audio_encoders.mert_projector import MERTProjector

    cpjku_dataset.load_piece = _patched_load_piece
    cpjku_audio_encoder.MERTProjector = MERTProjector
    print(f'[mert_patch] Patched load_piece + registered MERTProjector '
          f'(path_to_emb_root={_PATH_TO_EMB_ROOT})', flush=True)
