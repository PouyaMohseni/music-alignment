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
