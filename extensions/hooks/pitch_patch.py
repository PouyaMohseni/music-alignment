"""B2 -- monkey-patch the data pipeline to carry a per-timestep active-pitch
multi-hot vector (88-dim piano roll) alongside the existing perf/score/y
windows. None of the base pipeline's fields currently track pitch content
(only onset times), so this needs new data, unlike B3/B4/B5 which only need
new losses on already-available signals.

Two patches, both keyed off load_piece the same way B1a's MERT patch is
(module-level function/class reassignment, resolved fresh at call time by
load_dataset's own `ScoreAudioDataset(...)`/`load_piece` references, so it
also survives train_model.py's spawned multiprocessing workers):
  1. load_piece -> _patched_load_piece_pitch: unchanged CBEncoder spectrogram
     via the ORIGINAL load_performance, plus a NEW 'pitch_roll' key per
     tempo_factor computed from the same MIDI file.
  2. ScoreAudioDataset -> a subclass whose __getitem__ additionally windows
     and returns pitch_roll at the same per-timestep indexing already used
     for onsets/true_position (frame index i-pad, unpadded time).
"""
from __future__ import annotations
import copy

import numpy as np

NUM_PITCHES = 88
MIDI_PITCH_OFFSET = 21   # standard piano range 21-108 -> index 0-87


def _compute_pitch_roll(notes: np.ndarray, T_frames: int, fps: float) -> np.ndarray:
    """notes: (N, 5) [onset_sec, pitch, duration_sec, velocity, channel].
    Returns (T_frames, 88) float32 multi-hot active-pitch roll."""
    roll = np.zeros((T_frames, NUM_PITCHES), dtype=np.float32)
    onset_f = np.clip((notes[:, 0] * fps).astype(np.int64), 0, T_frames - 1)
    end_f = np.clip(((notes[:, 0] + notes[:, 2]) * fps).astype(np.int64), 0, T_frames)
    pitch_idx = np.clip((notes[:, 1] - MIDI_PITCH_OFFSET).astype(np.int64), 0, NUM_PITCHES - 1)
    for onset, end, p in zip(onset_f, end_f, pitch_idx):
        roll[onset:max(end, onset + 1), p] = 1.0
    return roll


def _patched_load_piece_pitch(params):
    """Reimplements audio_conditioned_unet.dataset.load_piece: unchanged
    CBEncoder spectrogram (via the real load_performance), plus pitch_roll
    computed from the same MIDI file."""
    import os
    from audio_conditioned_unet.utils import load_score, load_performance
    from madmom.io import midi as mm_midi

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
        spec, onsets, coords_new, interpol_fnc = load_performance(
            path, piece_name, spectrogram_params, coords, coord2onset,
            sf_path=params['sf_path'], tempo_factor=tempo_factor,
            real_perf=params['real_perf'], transpose=transpose)

        if tempo_factor == -1:
            midi_path = os.path.join(path, 'performance', piece_name + '.mid')
        else:
            midi_path = os.path.join(path, 'performance', piece_name + f'_tempo_{tempo_factor}.mid')
        midi = mm_midi.MIDIFile(midi_path)
        # spec is already padded by load_performance; pitch_roll indexed like
        # onsets (unpadded time), i.e. length = spec.shape[-1] - pad
        T_unpadded = spec.shape[-1] - spectrogram_params['pad']
        pitch_roll = _compute_pitch_roll(midi.notes, T_unpadded, spectrogram_params['fps'])

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


def patch_pitch_pipeline():
    from audio_conditioned_unet import dataset as cpjku_dataset

    cpjku_dataset.load_piece = _patched_load_piece_pitch
    cpjku_dataset.ScoreAudioDataset = PitchAwareScoreAudioDataset
    print('[pitch_patch] Patched load_piece + ScoreAudioDataset for B2 pitch roll', flush=True)


# Imported lazily inside patch_pitch_pipeline's caller context so the base
# ScoreAudioDataset class (needed as our base class) is importable first.
from audio_conditioned_unet.dataset import ScoreAudioDataset as _BaseScoreAudioDataset


class PitchAwareScoreAudioDataset(_BaseScoreAudioDataset):
    """Identical to the base __getitem__ except it also windows and returns
    pitch_roll at the same per-timestep indexing as onsets/true_position."""

    def __getitem__(self, item):
        if self.all_tempi:
            perf = self.all_perfs[item]['perf']
            score_id = self.all_perfs[item]['score_n']
            score = self.scores[score_id]
        else:
            score_id = item
            score = self.scores[item]
            perfs = self.performances[item]
            perf = perfs[np.random.choice(list(perfs.keys()))]

        spec = perf['spec']
        inp = perf['interpol_fnc']
        onsets = perf['onsets']
        pitch_roll = perf['pitch_roll']   # (T_unpadded, 88)

        scores = []
        perfs_out = []
        ys = []
        true_positions = []
        pitch_rolls = []

        max_y_shift = score.shape[0] - int(inp(spec.shape[-1])[0]) - 8   # MSMD_Y_OFFSET

        is_onset = []
        for i in range(self.pad, spec.shape[-1]):
            perfs_out.append(np.expand_dims(spec[:, i - self.n_frames + 1:i + 1], 0))

            true_position = np.array(inp(i - self.pad), dtype=np.int32)
            true_position, height = true_position[:-1], true_position[-1]

            y = np.zeros_like(score)
            y[true_position[0] - height // 2:true_position[0] + height // 2,
              true_position[1] - self.gt_width // 2:true_position[1] + self.gt_width // 2] = 1

            s = score
            yshift = xshift = 0
            if self.augment:
                yshift = np.random.randint(-9, max_y_shift)
                s = np.roll(score, yshift, 0)
                y = np.roll(y, yshift, 0)
                xshift = np.random.randint(-9, 13)
                s = np.roll(s, xshift, 1)
                y = np.roll(y, xshift, 1)

            ys.append(np.expand_dims(y, 0))
            scores.append(np.expand_dims(s, 0))
            true_positions.append(np.expand_dims(true_position, 0))

            frame_idx = i - self.pad
            is_onset.append(frame_idx in onsets)
            pr_idx = min(frame_idx, pitch_roll.shape[0] - 1)
            pitch_rolls.append(pitch_roll[pr_idx])

        perfs_out = np.concatenate(perfs_out)[:, np.newaxis]
        ys = np.concatenate(ys)[:, np.newaxis]
        scores = np.concatenate(scores)[:, np.newaxis]
        true_positions = np.concatenate(true_positions)
        pitch_rolls = np.stack(pitch_rolls).astype(np.float32)   # (T, 88)

        return {'inputs': {'perf': perfs_out, 'score': scores, 'length': scores.shape[0]},
                'targets': {'y': ys, 'true_positions': true_positions, 'pitch_roll': pitch_rolls},
                'file_name': self.piece_names[score_id],
                'interpol_c2o': perf['interpol_c2o'], 'add_per_staff': perf['add_per_staff'],
                'is_onset': is_onset}
