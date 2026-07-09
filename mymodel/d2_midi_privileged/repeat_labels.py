"""D2 component 1 -- MIDI-privileged repeat-aware soft CE labels.

Uses whole-piece MIDI (train-time only) to find score positions that are
musically indistinguishable from the true position at a given onset -- exact
or transposed local pitch-interval repeats -- so the training target need not
punish the model for landing on a genuinely ambiguous alternate. Nothing here
runs at inference: it only shapes the training target passed to
soft_multi_target_ce_loss (see losses.py); the similarity matrix, decode, and
eval pipeline are byte-for-byte D1's.
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np


def _interval_ngram_key(pitches: list[int]) -> tuple:
    """Transposition-invariant key: successive pitch DIFFERENCES, not absolute
    pitches, so a repeat transposed to a different key still matches."""
    return tuple(pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1))


def find_repeat_groups(note_onsets_frames: np.ndarray, note_pitches: np.ndarray,
                       onset_to_col: np.ndarray, k: int = 5) -> dict:
    """note_onsets_frames, note_pitches: (N,) parallel arrays, one entry per
    MIDI note, sorted by onset frame. onset_to_col: (N,) the score column
    (from coords) for each note -- the "true position" whose ambiguity we're
    checking. k: n-gram window length in notes.

    Returns {column: [alternate columns with an identical/transposed local
    pitch-interval context ending at that note]}. Only columns with >=2
    group members are included (i.e. genuine repeats) -- unique passages get
    an empty list and are unaffected (soft label degenerates to D1's original
    single-Gaussian target).
    """
    N = len(note_pitches)
    groups = defaultdict(list)   # ngram key -> list of note indices
    for i in range(k - 1, N):
        window = note_pitches[i - k + 1:i + 1].tolist()
        key = _interval_ngram_key(window)
        groups[key].append(i)

    col_alternates = defaultdict(set)
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        cols = [int(onset_to_col[i]) for i in idxs]
        for a in range(len(cols)):
            others = [cols[b] for b in range(len(cols)) if b != a]
            col_alternates[cols[a]].update(others)

    return {c: sorted(v) for c, v in col_alternates.items()}


def build_repeat_groups_for_piece(midi_path: str, coords: np.ndarray,
                                  onset_frames: np.ndarray, w_downsample: int,
                                  k: int = 5) -> dict:
    """coords: (N, >=2) [y, x, ...] strip px per onset (same array used to build
    interpol_fnc). onset_frames: (N,) frame per onset, same order as coords.
    Returns {column: [alternate columns]} as in find_repeat_groups."""
    import pretty_midi
    m = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted(((n.start, n.pitch) for inst in m.instruments for n in inst.notes),
                   key=lambda x: x[0])
    if len(notes) < k or len(coords) == 0:
        return {}

    # Match MIDI notes to the coords/onset_frames arrays by onset order. MSMD's
    # coords/onset_frames are already one-per-note in onset order (confirmed:
    # eval_official's own interpol_fnc construction assumes this), so simple
    # positional alignment holds as long as counts match reasonably closely.
    n = min(len(notes), len(coords), len(onset_frames))
    pitches = np.array([p for _, p in notes[:n]], dtype=np.int64)
    cols = np.round(coords[:n, 1] / w_downsample).astype(np.int64)

    return find_repeat_groups(onset_frames[:n], pitches, cols, k=k)
