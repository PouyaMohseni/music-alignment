"""E2/E3 -- MIDI-privileged signals ported to v13/v14/v15's dice/heatmap
paradigm. Reuses D2's repeat-detection and MidiEncoder machinery unchanged
(mymodel/d2_midi_privileged/{repeat_labels,midi_encoder}.py); this module only
adapts the INPUT format, since noteheads.npz already carries onset_sec/
midi_pitch/strip_x as parallel arrays (no separate MIDI-file parse needed,
unlike D2 which had to align a freshly-parsed MIDI file against coords/
onset_frames with a small note-count-mismatch risk -- that risk doesn't exist
here since these arrays come from the same annotation pipeline by construction).
"""
from __future__ import annotations
import numpy as np

from mymodel.d2_midi_privileged.repeat_labels import find_repeat_groups
from mymodel.d2_midi_privileged.midi_encoder import NUM_PITCHES, MIDI_PITCH_OFFSET


def build_repeat_alt_cols(onset_sec: np.ndarray, midi_pitch: np.ndarray,
                          strip_x: np.ndarray, w_scale: int, fps: int = 20,
                          k: int = 5) -> dict:
    """Returns {column: [alternate columns]} in STRIP-SCALED px (same space
    as strip_x_sc / make_gt_mask's cx), keyed by column -- exactly
    find_repeat_groups's return format, just fed pre-aligned parallel arrays
    instead of re-deriving them from a MIDI file."""
    order = np.argsort(onset_sec, kind='stable')
    onset_frames = np.round(onset_sec[order] * fps).astype(np.int64)
    pitches = midi_pitch[order].astype(np.int64)
    cols = np.round(strip_x[order] / w_scale).astype(np.int64)
    return find_repeat_groups(onset_frames, pitches, cols, k=k)


def compute_pitch_roll_from_notes(onset_sec: np.ndarray, offset_sec: np.ndarray,
                                  midi_pitch: np.ndarray, T: int, fps: int = 20) -> np.ndarray:
    """Same construction as D2's compute_pitch_roll, fed noteheads.npz's
    parallel arrays directly instead of parsing a MIDI file."""
    roll = np.zeros((T, NUM_PITCHES), dtype=np.float32)
    onset_f = np.clip(np.round(onset_sec * fps).astype(np.int64), 0, T - 1)
    end_f = np.clip(np.round(offset_sec * fps).astype(np.int64), 0, T)
    p = np.clip(midi_pitch.astype(np.int64) - MIDI_PITCH_OFFSET, 0, NUM_PITCHES - 1)
    for o, e, pi in zip(onset_f, end_f, p):
        roll[o:max(e, o + 1), pi] = 1.0
    return roll
