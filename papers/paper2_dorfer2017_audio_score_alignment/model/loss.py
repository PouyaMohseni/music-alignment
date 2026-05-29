"""
Pairwise ranking loss — Dorfer et al. 2017.
"""
import torch


def pairwise_ranking_loss(audio_emb: torch.Tensor,
                          score_emb: torch.Tensor,
                          margin: float = 0.1) -> torch.Tensor:
    """
    audio_emb: (B, D)  — L2 normalised
    score_emb: (B, D)  — i-th audio matches i-th score
    """
    sim = audio_emb @ score_emb.T          # (B, B) cosine similarity
    pos = torch.diag(sim)                  # matched pair scores

    loss_a2s = torch.clamp(margin - pos.unsqueeze(1) + sim, min=0).mean()
    loss_s2a = torch.clamp(margin - pos.unsqueeze(0) + sim, min=0).mean()
    return (loss_a2s + loss_s2a) / 2
