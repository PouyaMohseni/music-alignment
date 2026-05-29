"""
RL policy head — Henkel et al. 2019.
Actions: 0 = stay, 1 = advance in score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 256, n_actions: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, audio_emb: torch.Tensor, score_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([audio_emb, score_emb], dim=-1)
        return F.softmax(self.fc(x), dim=-1)  # action probabilities
