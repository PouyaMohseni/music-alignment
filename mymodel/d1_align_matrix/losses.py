"""D1 losses on the frame x column similarity matrix S (T, W_col).

- dense_ce_loss: per-frame cross-entropy toward a Gaussian target around the GT
  column. Primary supervision; monotonic GT columns implicitly teach a monotonic
  ridge. Softmax lives only inside the loss (never in a decoded output), so C1's
  attention-normalization geometry trap does not apply.
- banded_soft_dtw_loss: soft-DTW (Cuturi & Blondel 2017) over cost C = 1 - S,
  Sakoe-Chiba band-limited for tractability on large matrices. Path regularizer
  that trains the exact object the DTW decoder consumes. Guarded against the
  empty-reduction NaN that bit B4/C2.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_INF = 1e6   # finite stand-in for +inf in DP boundary cells (softmin-safe; see C2)


def dense_ce_loss(S: torch.Tensor, gt_cols: torch.Tensor, sigma_cols: float = 3.0,
                  valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """S: (T, W_col) similarity logits. gt_cols: (T,) long, GT column per frame.
    Gaussian-smoothed target over columns (label smoothing that respects the 1-D
    column geometry -- an off-by-one column should be nearly free). valid_mask:
    (T,) bool, frames to include (defaults to all)."""
    T, W = S.shape
    cols = torch.arange(W, device=S.device, dtype=torch.float32).unsqueeze(0)   # (1, W)
    centers = gt_cols.to(torch.float32).unsqueeze(1)                            # (T, 1)
    target = torch.exp(-0.5 * ((cols - centers) / sigma_cols) ** 2)             # (T, W)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    log_p = F.log_softmax(S, dim=-1)
    ce = -(target * log_p).sum(dim=-1)                                          # (T,)
    if valid_mask is not None:
        ce = ce[valid_mask]
        if ce.numel() == 0:
            return S.new_zeros(())
    return ce.mean()


def _downsample_matrix(S: torch.Tensor, max_t: int, max_w: int) -> torch.Tensor:
    """Adaptive-avg-pool S (T, W) down to at most (max_t, max_w), preserving the
    global path structure while keeping the soft-DTW DP small and fast."""
    T, W = S.shape
    out_t, out_w = min(T, max_t), min(W, max_w)
    x = S.unsqueeze(0).unsqueeze(0)                  # (1,1,T,W)
    x = F.adaptive_avg_pool2d(x, (out_t, out_w))
    return x.squeeze(0).squeeze(0)                   # (out_t, out_w)


def soft_dtw_matrix_loss(S: torch.Tensor, gamma: float = 0.1,
                         max_t: int = 200, max_w: int = 200) -> torch.Tensor:
    """Global soft-DTW (Cuturi & Blondel 2017) over cost C = -S (min-shifted to
    be >= 0) on a downsampled copy of the full similarity matrix. Trains the
    matrix so its globally-optimal monotonic path -- the exact object the DTW
    decoder extracts -- has low cost, complementing the local per-frame CE.

    Implemented with an anti-diagonal recursion so the DP is vectorized over each
    wavefront: only T'+W'-1 python iterations (~400 for 200x200), each a single
    tensor op -- no per-cell python loop, no NaN/inf (finite _INF sentinel, softmin
    via logsumexp). Returns cost normalized by path length.
    """
    if S.shape[0] == 0 or S.shape[1] == 0:
        return S.new_zeros(())
    C = _downsample_matrix(S, max_t, max_w)
    C = -C
    C = C - C.min().detach()          # shift to >=0, bounded; monotone in similarity
    n, m = C.shape

    # R[i,j] = soft-DTW accumulated cost; compute by anti-diagonals k = i+j.
    R = C.new_full((n + 1, m + 1), _INF)
    R[0, 0] = 0.0
    for k in range(2, n + m + 1):
        i_lo = max(1, k - m)
        i_hi = min(n, k - 1)
        if i_lo > i_hi:
            continue
        i_idx = torch.arange(i_lo, i_hi + 1, device=C.device)
        j_idx = k - i_idx
        r0 = R[i_idx - 1, j_idx]        # (i-1, j)
        r1 = R[i_idx - 1, j_idx - 1]    # (i-1, j-1)
        r2 = R[i_idx, j_idx - 1]        # (i, j-1)
        stacked = torch.stack([r0, r1, r2], dim=-1)             # (K, 3)
        softmin = -gamma * torch.logsumexp(-stacked / gamma, dim=-1)
        R = R.clone()
        R[i_idx, j_idx] = C[i_idx - 1, j_idx - 1] + softmin
    return R[n, m] / (n + m)
