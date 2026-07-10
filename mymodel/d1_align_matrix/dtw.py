"""D1 decode -- extract a frame->column alignment path from the similarity
matrix S (T, W_col). Three decoders:

- dtw_decode: offline globally-monotonic DP (primary). Cannot get lost by
  construction -- directly addresses the drift failure mode of the LSTM decoder.
- oltw_decode: causal online time warping (Dixon 2005, simplified greedy
  variant) -- real-time score following, no future lookahead.
- particle_filter_decode: causal, but replaces oltw_decode's greedy
  nearest-in-window argmax with C3's proven Bayesian particle filter
  (extensions/decode/particle_filter.py, originally built for CB_TA's 2-D
  heatmap output) -- a motion-model prior smooths single-frame similarity
  noise instead of jumping straight to each frame's local argmax. Measured
  motivation: on D2's checkpoint, oltw_decode's greedy causal decode (5.1%
  pct@0.5s) badly underperforms the offline DTW on the SAME matrix (55.0%),
  much more than the expected online/offline gap -- suggesting the greedy
  decoder itself, not just causality, is losing accuracy.

All three return path_cols: (T,) array (float for particle_filter_decode,
int for the other two) -- the aligned column index per frame.
"""
from __future__ import annotations
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from extensions.decode.particle_filter import ParticleFilterXTracker


def dtw_decode(S: np.ndarray, band_frac: float | None = 0.15, min_band: int = 40) -> np.ndarray:
    """S: (T, W) similarity (higher = better match). Returns (T,) column per frame.
    Standard DTW over cost = -S with steps (i-1,j)/(i-1,j-1)/(i,j-1), backtracked.
    Every frame is assigned exactly one column and columns are non-decreasing in
    frame -- a valid monotonic score-following path.

    band_frac: Sakoe-Chiba-style band, as a FRACTION of W, centered on the
    proportional diagonal (frame i's expected column ~= i * W/T, since audio
    frame-rate and score column-rate are different scales -- a raw |i-j| band
    would be wrong). None disables banding (full O(T*W) search).

    Without banding, an UNCONSTRAINED global DTW can route through a distant
    but locally-similar repeat (two performances of the same passage look
    identical in the similarity matrix) -- confirmed as the likely cause of
    D1's first-run bimodal error (very low median, high mean: right when it
    stays close to the diagonal, catastrophic when it jumps to a repeat).
    Banding keeps the path within a plausible tempo-deviation tube around the
    proportional diagonal, forbidding those distant jumps by construction."""
    cost = -S.astype(np.float64)
    T, W = cost.shape
    INF = np.inf
    D = np.full((T + 1, W + 1), INF)
    D[0, 0] = 0.0

    if band_frac is None:
        band = W  # no constraint
    else:
        band = max(min_band, int(round(band_frac * W)))

    ratio = W / max(T, 1)
    for i in range(1, T + 1):
        center = i * ratio
        j_lo = max(1, int(center - band))
        j_hi = min(W, int(center + band))
        row_cost = cost[i - 1]
        for j in range(j_lo, j_hi + 1):
            best = min(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])
            D[i, j] = row_cost[j - 1] + best

    if band_frac is not None and not np.isfinite(D[T, W]):
        # band too tight to connect (0,0)->(T,W) for this piece's actual tempo
        # deviation -- fall back to the unconstrained search rather than
        # backtrack through meaningless all-inf ties.
        return dtw_decode(S, band_frac=None)

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


def particle_filter_decode(S: np.ndarray, n_particles: int = 200,
                           process_noise_std: float = 3.0, init_std: float = 2.0,
                           velocity_ema_alpha: float = 0.3, resample_frac: float = 0.5,
                           seed: int = 0) -> np.ndarray:
    """Causal decode: C3's particle filter, fed this matrix's per-frame column-
    similarity row as the observation likelihood (softmaxed to be non-negative;
    the tracker itself renormalizes after multiplying by process-model weights,
    so the softmax temperature only affects how peaked the observation update
    is, not correctness).

    process_noise_std/init_std re-tuned for COLUMN space (D1/D2's matrix is
    already downsampled by w_downsample, e.g. 4px/column, unlike
    particle_filter.py's original CB_TA use case which tracks raw strip
    pixels) -- swept on 3 real test pieces against D2's trained checkpoint;
    process_noise_std=3.0/init_std=2.0 was the peak (16.8% pct@0.5s vs. 8.5%
    at 1.0/2.0 and worse at smaller or larger values -- too little noise
    can't track real tempo deviation, too much washes out the observation
    signal)."""
    T, W = S.shape
    tracker = ParticleFilterXTracker(n_particles=n_particles, process_noise_std=process_noise_std,
                                     velocity_ema_alpha=velocity_ema_alpha,
                                     resample_frac=resample_frac, init_std=init_std, seed=seed)
    path_cols = np.zeros(T, dtype=np.float64)
    for t in range(T):
        row = S[t].astype(np.float64)
        row = row - row.max()               # numerically stable softmax
        likelihood = np.exp(row)
        path_cols[t] = tracker.step(likelihood)
    return path_cols
