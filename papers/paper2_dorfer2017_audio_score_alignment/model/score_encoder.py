"""
Score image encoder — Dorfer et al. 2017.
Input : sheet music image patch (1, H, W)
Output: L2-normalised embedding (embedding_dim,)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        # TODO: match Table 1 in paper exactly
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 4 * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(self.conv(x).flatten(1))
        return F.normalize(x, dim=-1)
