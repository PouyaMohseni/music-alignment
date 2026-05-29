"""
Evaluation — Dorfer et al. 2017.
Metrics: Recall@1, Recall@10 on MSMD test set.
"""
import torch


def recall_at_k(audio_embs: torch.Tensor,
                score_embs: torch.Tensor,
                k: int = 1) -> float:
    """
    audio_embs: (N, D) — L2 normalised
    score_embs: (N, D) — i-th audio matches i-th score
    """
    n = audio_embs.size(0)
    sim = audio_embs @ score_embs.T
    ranks = sim.argsort(dim=-1, descending=True)
    correct = (ranks[:, :k] == torch.arange(n).unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()
