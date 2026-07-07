"""C2 -- Differentiable Soft-DTW monotonic-alignment loss (Cuturi & Blondel 2017).

Unlike B4's temporal_consistency_loss (plain L1 + monotonicity/jerk penalties,
comparing predicted and GT positions strictly frame-by-frame at matching
timestep indices), soft-DTW compares the predicted trajectory against the GT
trajectory under the best monotonic warping -- so a predicted trajectory
that's a few frames early/late (a lag the RNN's own dynamics introduce, not a
real localization error) isn't penalized as harshly as a same-index L1 term
would, while still requiring the overall path shape to match.
"""
from __future__ import annotations
import torch

# Sentinel standing in for +infinity in the DP boundary cells. Large enough
# that softmin's exp(-sentinel/gamma) underflows cleanly to 0 in float32
# without ever touching a real inf/nan in the recursion (gamma is O(0.1),
# so -sentinel/gamma is already ~-1e7, safely beyond underflow).
_INF_COST = 1e6


def _pairwise_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: (B, T1, D), y: (B, T2, D) -> (B, T1, T2) squared Euclidean distances."""
    x2 = (x ** 2).sum(-1, keepdim=True)                    # (B, T1, 1)
    y2 = (y ** 2).sum(-1, keepdim=True).transpose(1, 2)    # (B, 1, T2)
    xy = torch.bmm(x, y.transpose(1, 2))                    # (B, T1, T2)
    return (x2 + y2 - 2 * xy).clamp_min(0.0)


def soft_dtw_loss(pred_positions: torch.Tensor, gt_positions: torch.Tensor,
                   gamma: float = 0.1) -> torch.Tensor:
    """pred_positions, gt_positions: (T, B, 2) -- [x, y] per BPTT timestep,
    same (seq_len, batch) layout as B4's temporal_consistency_loss (NOT
    (B, T, 2)). Positions should already be normalized to [0, 1] by the
    caller (same discipline as B4 -- see c2_callback.py), so the squared-
    distance cost matrix is O(1) scale, comparable to dice loss.

    Returns the mean (over batch) soft-DTW alignment cost, normalized by
    trajectory length T so BPTT chunks of different lengths (the last chunk
    of a piece can be shorter than seq_len) contribute a comparable scale.

    Uses a python list-of-lists DP grid (T is small, <= ~16) rather than a
    preallocated tensor mutated in-place via indexed assignment -- avoids any
    ambiguity around in-place ops inside the autograd graph.
    """
    T, B, _ = pred_positions.shape
    if T == 0:
        return pred_positions.new_zeros(())

    x = pred_positions.permute(1, 0, 2)   # (B, T, D)
    y = gt_positions.permute(1, 0, 2)     # (B, T, D)
    C = _pairwise_sqdist(x, y)             # (B, T, T)

    inf_cost = C.new_full((B,), _INF_COST)
    zero = C.new_zeros((B,))

    R = [[None] * (T + 1) for _ in range(T + 1)]
    R[0][0] = zero
    for i in range(1, T + 1):
        R[i][0] = inf_cost
    for j in range(1, T + 1):
        R[0][j] = inf_cost

    for i in range(1, T + 1):
        for j in range(1, T + 1):
            r = torch.stack([R[i - 1][j], R[i - 1][j - 1], R[i][j - 1]], dim=-1)   # (B, 3)
            softmin = -gamma * torch.logsumexp(-r / gamma, dim=-1)
            R[i][j] = C[:, i - 1, j - 1] + softmin

    return (R[T][T] / T).mean()
