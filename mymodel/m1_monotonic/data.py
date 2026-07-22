"""M1 data helper. Derives the onset-column representation the forward-sum
objective needs from a D1Piece (mymodel/d1_align_matrix/data.py's load_piece,
which we reuse verbatim for the strip/MERT/geometry loading -- no duplication).

From a D1Piece we build, for the (optionally frame-truncated) piece:
  - onset_frames   (N,) : the piece's unique onset frames within [0, T)
  - onset_cols     (N,) : spatial-column index of each onset (= gt_cols at that
                          frame), the index into the score tower's W_col output
  - onset_x        (N,) : strip-x pixel of each onset column (for position readout)
  - gt_path        (T,) : GT onset-column index per frame = index of the most
                          recent onset at/ before each frame, clamped to [0,N-1].
                          This is monotone, stay-or-advance-by-one, surjective,
                          starts at 0, ends at N-1 -- exactly forward-sum's path
                          model (see extensions/alignment/forward_sum.py).
"""
from __future__ import annotations
import numpy as np


def build_onset_columns(piece, t_max: int | None = None):
    """Returns (T, onset_frames, onset_cols, onset_x, gt_path) as numpy arrays,
    or None if the (truncated) piece has no usable onsets. T is the (possibly
    truncated) frame count; all frame indices are guaranteed < T and T >= N."""
    gt_cols = piece.gt_cols.numpy().astype(np.int64)      # (T_full,)
    T = len(gt_cols)
    if t_max is not None:
        T = min(T, t_max)

    onset_frames = np.asarray(piece.onset_frames, dtype=np.int64)
    onset_frames = np.unique(onset_frames[(onset_frames >= 0) & (onset_frames < T)])
    N = len(onset_frames)
    if N < 2 or T < N:
        return None

    onset_cols = gt_cols[onset_frames]                    # (N,) spatial column per onset
    wd = piece.w_downsample
    onset_x = onset_cols.astype(np.float64) * wd + wd / 2.0

    # most-recent-onset index per frame; frames before the first onset -> 0.
    gt_path = np.searchsorted(onset_frames, np.arange(T), side='right') - 1
    gt_path = np.clip(gt_path, 0, N - 1).astype(np.int64)

    return T, onset_frames, onset_cols, onset_x, gt_path
