"""Merged load_piece patch for MERT+B2: B1a's frozen-MERT load_piece patch
(mert_patch.py) and B2's pitch-roll load_piece patch (pitch_patch.py) BOTH
override audio_conditioned_unet.dataset.load_piece -- applying both patches
naively means whichever one patches last silently wins, with no error to
catch it (confirmed by reading both files before writing this). This module
properly merges them into ONE load_piece that does both: MERT embedding
lookup as 'spec' (instead of a live CBEncoder spectrogram) AND pitch_roll
computed from the same MIDI file already loaded for onset extraction.

PitchAwareScoreAudioDataset (pitch_patch.py) needs NO changes -- its
__getitem__ only reads perf['pitch_roll'] and perf['spec'] generically,
never assuming what produced 'spec', so it's reused unchanged here.
"""
from __future__ import annotations
import copy

import numpy as np

from extensions.hooks.mert_patch import _load_mert_spec
from extensions.hooks.pitch_patch import _compute_pitch_roll, PitchAwareScoreAudioDataset


def _patched_load_performance_mert_pitch(path, piece, spectrogram_params, coords, coord2onset,
                                         sf_path, tempo_factor=1., real_perf=False, transpose=0):
    """Same structure as mert_patch._patched_load_performance, but also
    computes pitch_roll from the same MIDI file (reusing its already-loaded
    midi.notes instead of reloading the file a second time, unlike a naive
    two-patch stack would)."""
    import os
    from madmom.io import midi as mm_midi
    from scipy import interpolate
    from audio_conditioned_unet.utils import merge_onsets

    if real_perf:
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

    spec = _load_mert_spec(path, piece, tempo_factor)
    spec = np.pad(spec, ((0, 0), (spectrogram_params['pad'], 0)), mode='constant')

    T_unpadded = spec.shape[-1] - spectrogram_params['pad']
    pitch_roll = _compute_pitch_roll(midi.notes, T_unpadded, spectrogram_params['fps'])

    onsets = (midi.notes[:, 0] * spectrogram_params['fps']).astype(int)
    onsets, coords_new = merge_onsets(onsets, copy.deepcopy(coords), coord2onset[0])
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return spec, onsets, coords_new, interpol_fnc, pitch_roll


def _patched_load_piece_mert_pitch(params):
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
        spec, onsets, coords_new, interpol_fnc, pitch_roll = _patched_load_performance_mert_pitch(
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
                                      'add_per_staff': [staff_coords, add_per_staff],
                                      'pitch_roll': pitch_roll,
                                      }

    return i, score, piece_name, performances


def patch_mert_pitch_pipeline(path_to_emb_root: dict[str, str]):
    """Call once, before load_dataset()/ConditionalUNet() -- registers
    MERTProjector, replaces load_piece with the merged MERT+pitch-roll
    version, and replaces ScoreAudioDataset with the pitch-aware one
    (unchanged from pitch_patch.py -- generic over what 'spec' contains)."""
    import os
    from extensions.hooks import mert_patch as _mp
    _mp._PATH_TO_EMB_ROOT = dict(path_to_emb_root)
    os.environ['MERT_PATH_MAP'] = ';'.join(f'{k}={v}' for k, v in path_to_emb_root.items())

    from audio_conditioned_unet import dataset as cpjku_dataset
    from audio_conditioned_unet import audio_encoder as cpjku_audio_encoder
    from extensions.audio_encoders.mert_projector import MERTProjector

    cpjku_dataset.load_piece = _patched_load_piece_mert_pitch
    cpjku_dataset.ScoreAudioDataset = PitchAwareScoreAudioDataset
    cpjku_audio_encoder.MERTProjector = MERTProjector
    print(f'[mert_pitch_patch] Patched load_piece (MERT audio + pitch_roll) + '
          f'ScoreAudioDataset + registered MERTProjector '
          f'(path_to_emb_root={path_to_emb_root})', flush=True)
