"""D1 -- two-tower audio/score encoders producing a frame x column similarity
matrix, decoded by monotonic DTW. See D1.md for the full rationale.

Audio tower: precomputed MERT (T, 768) -> small temporal conv encoder -> (T, d),
L2-normalized per frame.
Score tower: strip image (1, H, W) -> CNN (width strided by w_downsample, height
fully pooled) -> per-column features -> transformer over columns -> (W_col, d),
L2-normalized per column. Computed once per piece.
Similarity: S = A @ B.T / temperature, shape (T, W_col).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioTower(nn.Module):
    def __init__(self, d_mert: int = 768, d_model: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(d_mert, d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(1, d_model)
        self.norm2 = nn.GroupNorm(1, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, mert: torch.Tensor) -> torch.Tensor:
        """mert: (T, d_mert) -> (T, d_model) L2-normalized per frame."""
        x = self.in_proj(mert)               # (T, d)
        x = x.transpose(0, 1).unsqueeze(0)   # (1, d, T) for Conv1d
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = x.squeeze(0).transpose(0, 1)     # (T, d)
        x = self.out_proj(x)
        return F.normalize(x, dim=-1)


def _sinusoidal_pe(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe


class ScoreTower(nn.Module):
    def __init__(self, d_model: int = 128, w_downsample: int = 4,
                 n_ctx_layers: int = 2, n_heads: int = 4):
        super().__init__()
        assert w_downsample in (2, 4, 8), 'w_downsample must be a power of 2 in {2,4,8}'
        self.w_downsample = w_downsample
        n_wstride_2 = int(round(math.log2(w_downsample)))   # how many stride-2-in-W blocks

        chans = [1, 32, 64, d_model]
        blocks = []
        for i in range(3):
            # stride width by 2 for the first n_wstride_2 blocks; height strided by 2
            # every block (height is small, gets pooled to 1 at the end anyway).
            w_stride = 2 if i < n_wstride_2 else 1
            blocks.append(nn.Conv2d(chans[i], chans[i + 1], kernel_size=3,
                                    stride=(2, w_stride), padding=1))
            blocks.append(nn.GroupNorm(1, chans[i + 1]))
            blocks.append(nn.GELU())
        self.cnn = nn.Sequential(*blocks)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                               batch_first=True, activation='gelu')
        self.ctx = nn.TransformerEncoder(enc_layer, num_layers=n_ctx_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, strip: torch.Tensor) -> torch.Tensor:
        """strip: (1, 1, H, W) -> (W_col, d_model) L2-normalized per column.
        W_col = ceil(W / w_downsample)."""
        x = self.cnn(strip)                       # (1, d, H', W_col)
        x = x.mean(dim=2)                          # (1, d, W_col) -- pool height to 1
        x = x.squeeze(0).transpose(0, 1)           # (W_col, d)
        x = x + _sinusoidal_pe(x.shape[0], self.d_model, x.device)
        x = self.ctx(x.unsqueeze(0)).squeeze(0)    # (W_col, d)
        x = self.out_proj(x)
        return F.normalize(x, dim=-1)


class D1Model(nn.Module):
    def __init__(self, d_mert: int = 768, d_model: int = 128, w_downsample: int = 4,
                 n_ctx_layers: int = 2, n_heads: int = 4, temperature: float = 0.07):
        super().__init__()
        self.audio_tower = AudioTower(d_mert, d_model)
        self.score_tower = ScoreTower(d_model, w_downsample, n_ctx_layers, n_heads)
        self.temperature = temperature
        self.w_downsample = w_downsample

    def encode(self, mert: torch.Tensor, strip: torch.Tensor):
        return self.audio_tower(mert), self.score_tower(strip)

    def similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A: (T, d), B: (W_col, d) -> S: (T, W_col)."""
        return (A @ B.t()) / self.temperature

    def forward(self, mert: torch.Tensor, strip: torch.Tensor) -> torch.Tensor:
        A, B = self.encode(mert, strip)
        return self.similarity(A, B)
