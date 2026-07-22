"""Repeat unrolling -- the crux design point that makes forward-sum applicable
to score-following with repeats.

A printed repeat is drawn ONCE but played TWICE, so the true alignment path is
monotone in audio-time but SAWTOOTHS in printed-column space (forward, jump
back to the repeat start, forward again). Vanilla forward-sum/Viterbi
(forward_sum.py, monotonic_decode.py) assume the path is monotone in BOTH
axes, so they cannot be applied directly to printed columns.

The fix: align audio to the UNROLLED score -- a longer virtual-column sequence
in which each played pass appears explicitly, so the path is genuinely
monotone in (audio-time, virtual-column). Each virtual column carries a
back-pointer to the printed column it renders, so a virtual alignment folds
back onto the printed strip for readout/visualisation, and a virtual posterior
can be summed onto printed columns.

This module is the representational bridge from the existing repeat
infrastructure (mymodel/d2_midi_privileged/repeat_labels.find_repeat_groups,
which represents ambiguity as {printed_col: [alternate printed_cols]}) to the
sequence representation forward-sum needs. Scope: non-nested, non-overlapping
repeat spans, each played twice -- the common MSMD case. Nested repeats /
voltas are a Phase-1+ extension (the virtual_to_printed contract below does
not change, only how the sequence is constructed).
"""
from __future__ import annotations
import numpy as np


def unroll_repeats(n_cols: int, repeat_spans: list[tuple[int, int]]) -> np.ndarray:
    """n_cols: number of printed columns [0, n_cols). repeat_spans: list of
    (start, end) INCLUSIVE printed-column ranges, each played twice, given in
    left-to-right order, non-nested and non-overlapping.

    Returns virtual_to_printed: (V,) int array mapping each virtual column to
    its printed column. Walking virtual columns 0..V-1 in order reproduces the
    played trajectory: for a span (s, e), columns s..e appear, then s..e again,
    then the score continues at e+1.

    Example: n_cols=10, repeat_spans=[(3, 6)] ->
        [0,1,2, 3,4,5,6, 3,4,5,6, 7,8,9]
    """
    spans = sorted(repeat_spans)
    for i in range(1, len(spans)):
        if spans[i][0] <= spans[i - 1][1]:
            raise ValueError(f"overlapping/nested repeat spans not supported in Phase 0: {spans}")

    virtual = []
    n = 0
    si = 0
    while n < n_cols:
        virtual.append(n)
        if si < len(spans) and n == spans[si][1]:
            s, e = spans[si]
            virtual.extend(range(s, e + 1))   # the repeated pass
            si += 1
        n += 1
    return np.asarray(virtual, dtype=np.int64)


def printed_path_from_virtual(virtual_path: np.ndarray,
                              virtual_to_printed: np.ndarray) -> np.ndarray:
    """Fold a virtual-column path (monotone) back to printed columns (sawtooth).
    virtual_path: (T,) virtual column indices. Returns (T,) printed columns."""
    return virtual_to_printed[virtual_path]


def fold_posterior_to_printed(virtual_posterior: np.ndarray,
                              virtual_to_printed: np.ndarray, n_cols: int) -> np.ndarray:
    """Sum a (T, V) virtual-column posterior onto (T, n_cols) printed columns,
    so a virtual alignment posterior can be visualised on the printed strip.
    Two virtual columns rendering the same printed column add their mass."""
    T = virtual_posterior.shape[0]
    printed = np.zeros((T, n_cols), dtype=virtual_posterior.dtype)
    np.add.at(printed.T, virtual_to_printed, virtual_posterior.T)
    return printed


def unroll_column_x(col_x: np.ndarray, virtual_to_printed: np.ndarray) -> np.ndarray:
    """Map printed-column x-coordinates (n_cols,) to virtual-column
    x-coordinates (V,) -- so a virtual path decodes straight to strip x."""
    return col_x[virtual_to_printed]
