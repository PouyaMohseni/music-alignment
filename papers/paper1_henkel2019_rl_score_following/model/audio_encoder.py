"""
Audio encoder — Henkel et al. 2019.
Input : log-filterbank spectrogram (1, n_bins, T)
Output: embedding vector (embedding_dim,)
"""
import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self, n_bins: int = 92, embedding_dim: int = 256):
        super().__init__()
        # TODO: match exact architecture from paper (Section 3)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(64 * 4 * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))
