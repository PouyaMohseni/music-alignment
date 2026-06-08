"""SoftDTW loss + anchor regulariser + optional Sakoe-Chiba band mask."""
from __future__ import annotations
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- band mask ---


def sakoe_chiba_mask(T: int, N: int, radius_frac: float, device, dtype) -> torch.Tensor:
    """Return a (T, N) tensor that is 0 inside a diagonal band of half-width
    radius = ceil(max(T, N) * radius_frac), and a large finite penalty outside.

    The penalty is finite (not +inf) so SoftDTW's soft-min stays well behaved;
    in practice 1e4 is enough to make off-band paths effectively impossible.
    """
    radius = max(1, int((max(T, N) * radius_frac)))
    # diagonal slope: each row's optimal column is at j ≈ i * (N-1)/(T-1)
    i = torch.arange(T, device=device).float().view(T, 1)
    j = torch.arange(N, device=device).float().view(1, N)
    proj_j = i * ((N - 1) / max(T - 1, 1))
    inside = (j - proj_j).abs() <= radius
    mask = torch.zeros(T, N, device=device, dtype=dtype)
    mask[~inside] = 1.0e4
    return mask


# ------------------------------------------------------------------- losses ---


def _gather_anchor_sim(
    sim: torch.Tensor,                # (B, T, N)
    anchors_t: torch.Tensor,           # (B, K)
    anchors_n: torch.Tensor,           # (B, K)
) -> torch.Tensor:
    B, T, N = sim.shape
    t_safe = anchors_t.clamp(min=0, max=T - 1)
    n_safe = anchors_n.clamp(min=0, max=N - 1)
    flat_idx = t_safe * N + n_safe                      # (B, K)
    sim_flat = sim.reshape(B, T * N)                    # (B, T*N)
    return sim_flat.gather(1, flat_idx)                 # (B, K)


def softdtw_anchor_loss(
    sim: torch.Tensor,                 # (B, T, N) cosine similarity in [-1, 1]
    anchors_t: torch.Tensor,           # (B, K) int64, -1 padding
    anchors_n: torch.Tensor,           # (B, K) int64, -1 padding
    anchor_mask: torch.Tensor,         # (B, K) bool
    *,
    gamma: float = 0.1,
    anchor_weight: float = 1.0,
    band_radius_frac: float | None = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SoftDTW(1 - sim) plus an MSE pull on anchor cells.

    Returns
    -------
    loss : scalar tensor
    parts : dict[str, scalar tensor]   logged components
    """
    B, T, N = sim.shape
    cost = 1.0 - sim                                    # (B, T, N) in [0, 2]

    if band_radius_frac is not None:
        band = sakoe_chiba_mask(T, N, band_radius_frac, sim.device, cost.dtype)
        cost = cost + band                              # broadcasts over B

    # pysdtw's SoftDTW expects two sequences and an internal distance, but it
    # also accepts a precomputed cost matrix via the `D` argument-less API by
    # subclassing. The package's stable interface uses SoftDTW(D=...) so we
    # construct a tiny adapter via SoftDTWLossPyTorch's _SoftDTW class.
    # The simpler portable route: call pysdtw.SoftDTW with explicit pairwise.
    # For our case we build SoftDTW from scratch on the cost matrix.
    L_dtw = _softdtw_from_cost(cost, gamma=gamma)
    L_dtw = L_dtw.mean()

    # Anchor pull: cells (t_k, n_k) should be similar (sim ≈ 1)
    sim_a = _gather_anchor_sim(sim, anchors_t, anchors_n)        # (B, K)
    sq = (1.0 - sim_a) ** 2
    denom = anchor_mask.float().sum().clamp(min=1.0)
    L_anchor = (sq * anchor_mask.float()).sum() / denom

    loss = L_dtw + anchor_weight * L_anchor
    return loss, {"dtw": L_dtw.detach(), "anchor": L_anchor.detach()}


# ------------------------------------- SoftDTW on a precomputed cost matrix ---


def _softdtw_from_cost(cost: torch.Tensor, gamma: float) -> torch.Tensor:
    """SoftDTW on a (B, T, N) cost matrix — pure-PyTorch recursion.

    Training windows are small (T~50, N~7) so the Python loop is fast enough.
    pysdtw's API does not cleanly accept a precomputed cost matrix, so we avoid
    it entirely to prevent negative-loss bugs.
    """
    B, T, N = cost.shape
    r = cost.new_full((B, T + 1, N + 1), float("inf"))
    r[:, 0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, N + 1):
            triple = torch.stack([r[:, i - 1, j - 1],
                                  r[:, i - 1, j],
                                  r[:, i, j - 1]], dim=-1)
            softmin = -gamma * torch.logsumexp(-triple / gamma, dim=-1)
            r[:, i, j] = cost[:, i - 1, j - 1] + softmin
    return r[:, T, N]
