"""
Loss functions shared across model versions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dtw_loss(pred_path: torch.Tensor,
                  target_path: torch.Tensor,
                  gamma: float = 1.0) -> torch.Tensor:
    """
    SoftDTW — differentiable DTW that respects monotonic alignment.
    TODO: use sdtw_cuda or tslearn.metrics.SoftDTW.
    Reference: Cuturi & Blondel, NeurIPS 2017.
    """
    raise NotImplementedError("Install tslearn or sdtw_cuda, then implement here")


def infonce_loss(audio_emb: torch.Tensor,
                 score_emb: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss (used as auxiliary in v3)."""
    a = F.normalize(audio_emb, dim=-1)
    s = F.normalize(score_emb, dim=-1)
    logits = a @ s.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return F.cross_entropy(logits, labels)


def heatmap_inr_loss(
    confidence: torch.Tensor,   # (T, Q) raw INR logits over continuous queries
    query_x: torch.Tensor,      # (Q,) pixel position of each query
    pos_target: torch.Tensor,   # (T,) ground-truth pixel position
    valid_mask: torch.Tensor,   # (T,) bool
    sigma_px: float = 20.0,
) -> tuple[torch.Tensor, dict]:
    """Heatmap regression loss for the M06 INR head.

    Target for each valid frame is a Gaussian centered at the true pixel
    position (width sigma_px), normalized to a distribution over the query
    grid; loss is the soft cross-entropy between that target and
    softmax(confidence). Continuous queries (not tied to any discrete tile
    grid) are what let this break the M01-M05 tile-quantization ceiling —
    resolution is set by sigma_px and query density, not by column spacing.
    """
    T, Q = confidence.shape
    if valid_mask.sum() == 0:
        z = confidence.sum() * 0.0
        return z, {"heatmap_ce": z.detach()}

    diff = (query_x.view(1, Q) - pos_target.view(T, 1)) / sigma_px
    target = torch.exp(-0.5 * diff ** 2)                      # (T, Q)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    log_p = F.log_softmax(confidence, dim=-1)                  # (T, Q)
    ce = -(target * log_p).sum(dim=-1)                          # (T,)

    m = valid_mask.float()
    denom = m.sum().clamp(min=1.0)
    loss = (ce * m).sum() / denom
    return loss, {"heatmap_ce": loss.detach()}


def expected_distance_loss(
    sim: torch.Tensor,          # (T, N) similarity, audio frame × strip tile
    pos_tile: torch.Tensor,     # (N,) normalized tile position in [0, 1]
    pos_target: torch.Tensor,   # (T,) normalized ground-truth position in [0, 1]
    valid_mask: torch.Tensor,   # (T,) bool — frames with a known target
    *,
    temperature: float = 0.07,
    power: float = 1.0,         # 1.0 = L1 distance, 2.0 = squared
    entropy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Distance-aware soft-localization loss over a full performance.

    For each audio frame t, the model's distribution over strip tiles is
        p[t, :] = softmax(sim[t, :] / temperature)
    and the loss is the EXPECTED distance from the predicted tile to the true
    position:
        L(t) = sum_n p[t, n] * |pos_tile[n] - pos_target[t]| ** power

    Near misses cost little; far misses cost in proportion to distance, and a
    bimodal "hedge" can't cheat (any mass placed far away raises the expectation).
    This replaces both DTW and InfoNCE for alignment training.

    Returns (loss, parts) where parts logs the mean expected distance and the
    mean entropy of the predicted distributions (for monitoring collapse).
    """
    T, N = sim.shape
    if valid_mask.sum() == 0:
        z = sim.sum() * 0.0
        return z, {"exp_dist": z.detach(), "entropy": z.detach()}

    p = F.softmax(sim / temperature, dim=-1)                 # (T, N)
    # distance from every tile to each frame's target: (T, N)
    dist = (pos_tile.view(1, N) - pos_target.view(T, 1)).abs()
    if power != 1.0:
        dist = dist ** power
    exp_dist = (p * dist).sum(dim=-1)                        # (T,)

    m = valid_mask.float()
    denom = m.sum().clamp(min=1.0)
    loss = (exp_dist * m).sum() / denom

    # optional entropy regulariser — encourage peaked (not uniform) distributions
    entropy = -(p * (p.clamp_min(1e-9)).log()).sum(dim=-1)   # (T,)
    mean_entropy = (entropy * m).sum() / denom
    if entropy_weight > 0:
        loss = loss + entropy_weight * mean_entropy

    return loss, {"exp_dist": ((exp_dist * m).sum() / denom).detach(),
                  "entropy": mean_entropy.detach()}
