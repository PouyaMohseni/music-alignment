"""D1 decode -- extract a frame->column alignment path from the similarity
matrix S (T, W_col). Two decoders:

- dtw_decode: offline globally-monotonic DP (primary). Cannot get lost by
  construction -- directly addresses the drift failure mode of the LSTM decoder.
- oltw_decode: causal online time warping (Dixon 2005) -- real-time score
  following, no future lookahead, for an honest online-tracking claim.

Both return path_cols: (T,) int array, the aligned column index per frame.
"""
from __future__ import annotations
import numpy as np


def dtw_decode(S: np.ndarray) -> np.ndarray:
    """S: (T, W) similarity (higher = better match). Returns (T,) column per frame.
    Standard DTW over cost = -S with steps (i-1,j)/(i-1,j-1)/(i,j-1), backtracked.
    Every frame is assigned exactly one column and columns are non-decreasing in
    frame -- a valid monotonic score-following path."""
    cost = -S.astype(np.float64)
    T, W = cost.shape
    INF = np.inf
    D = np.full((T + 1, W + 1), INF)
    D[0, 0] = 0.0
    # allow starting anywhere in the first few columns (piece may not start at col 0
    # after strip padding); permit column 0..W-1 as valid starts by seeding row 0's
    # accumulation along columns is NOT wanted (that would let the path skip columns
    # for free). Instead seed only (0,0); the band is wide enough via steps.
    for i in range(1, T + 1):
        row_cost = cost[i - 1]
        for j in range(1, W + 1):
            best = min(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])
            D[i, j] = row_cost[j - 1] + best

    # backtrack from (T, W)
    i, j = T, W
    path_cols = np.zeros(T, dtype=np.int64)
    while i > 0:
        path_cols[i - 1] = j - 1
        if j == 0:
            i -= 1
            continue
        up, diag, left = D[i - 1, j], D[i - 1, j - 1], D[i, j - 1]
        m = min(up, diag, left)
        if m == diag:
            i, j = i - 1, j - 1
        elif m == up:
            i, j = i - 1, j
        else:
            j = j - 1
    return np.clip(path_cols, 0, W - 1)


def oltw_decode(S: np.ndarray, search: int = 40) -> np.ndarray:
    """Causal online time warping (Dixon 2005, simplified). For each frame in
    order, advance the current column pointer to the best-matching column within
    a forward-only search window [cur, cur+search], never moving backward.
    Returns (T,) column per frame. No future frames are consulted -> real-time
    score-following semantics."""
    T, W = S.shape
    path_cols = np.zeros(T, dtype=np.int64)
    cur = 0
    for t in range(T):
        hi = min(W, cur + search + 1)
        window = S[t, cur:hi]
        if window.size == 0:
            path_cols[t] = cur = W - 1
            continue
        best_local = int(np.argmax(window))
        cur = cur + best_local          # monotonic: cur never decreases
        path_cols[t] = cur
    return path_cols
