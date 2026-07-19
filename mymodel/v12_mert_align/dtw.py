"""
Standard DTW for alignment decoding.
Input: cost matrix (T_audio, N_cols); output: path array of length T_audio.
Pure-PyTorch, runs on CPU (inference only).
"""
import numpy as np
import torch


def dtw_decode(sim: torch.Tensor) -> np.ndarray:
    """
    sim: (T, N) similarity matrix (higher = better match).
    Returns path: (T,) int array mapping each audio frame to a column index.
    Enforces monotonicity and continuity (steps: stay, right, diagonal).
    """
    cost = (1.0 - sim.cpu().float().numpy())   # lower = better
    T, N = cost.shape

    D  = np.full((T, N), np.inf)
    bt = np.zeros((T, N), dtype=np.int16)   # backtrack: 0=diag,1=left,2=up

    D[0, 0] = cost[0, 0]
    for j in range(1, N):
        D[0, j] = D[0, j - 1] + cost[0, j]
        bt[0, j] = 1
    for i in range(1, T):
        D[i, 0] = D[i - 1, 0] + cost[i, 0]
        bt[i, 0] = 2

    for i in range(1, T):
        for j in range(1, N):
            candidates = [D[i-1, j-1], D[i-1, j], D[i, j-1]]
            best = int(np.argmin(candidates))
            D[i, j] = candidates[best] + cost[i, j]
            bt[i, j] = best

    # Backtrack. Forward-pass candidate order is [diag, up, left] (line 32:
    # candidates = [D[i-1,j-1], D[i-1,j], D[i,j-1]]) -- so bt==1 means the
    # predecessor is D[i-1,j] (same column, previous row: "up", i decreases),
    # and bt==2 means D[i,j-1] (same row, previous column: "left", j
    # decreases). This was previously swapped, making dtw_decode() return a
    # near-model-independent degenerate path at realistic T>>N scale
    # (confirmed empirically -- see mymodel/v12_mert_align audit).
    path = np.zeros(T, dtype=np.int32)
    i, j = T - 1, N - 1
    while i > 0 or j > 0:
        path[i] = j
        b = bt[i, j]
        if b == 0:
            i -= 1; j -= 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    path[0] = j
    return path
