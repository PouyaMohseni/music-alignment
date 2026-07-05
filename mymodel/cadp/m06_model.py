"""CADP M06 — INR Head (continuous position).

Purpose: break the tile-quantization resolution ceiling that every prior
model (M01-M05) inherits from decoding onto a fixed discrete grid (score
columns, or their x4 sub-positions). Instead of a per-cell logit over a fixed
set of positions, an implicit neural representation (INR) takes a per-frame
condition vector plus ANY continuous query pixel coordinate and outputs a
confidence — so position resolution is limited only by query density and the
supervision's Gaussian width (sigma_px), not by how the score was tiled.

Architecture:
  M04DenseTokens encoder -> dense sim matrix -> conv_stem + axial attention
  (identical to M05's PathPredictor backbone) -> per-cell features (T_a,N_s,C)
  cond_vec = Linear(C, cond_dim)(mean_over_score_axis(features))  # (T_a, cond_dim)
  INRHead(cond_vec, query_x) -> confidence (T_a, Q) for continuous query_x
  predicted position = soft-argmax(confidence, query_x)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mymodel.cadp.m04_model import M04DenseTokens
from mymodel.cadp.m05_model import AxialAttentionBlock


class INRHead(nn.Module):
    def __init__(self, cond_dim: int = 64, hidden_dim: int = 256,
                 fourier_freqs=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.register_buffer('freqs', torch.tensor(fourier_freqs, dtype=torch.float32))
        in_dim = len(fourier_freqs) * 2 + cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def fourier_encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: (Q,) in [0,1] -> (Q, 2*len(freqs))."""
        angles = x_norm.view(-1, 1) * self.freqs.view(1, -1) * torch.pi   # (Q, F)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)   # (Q, 2F)

    def forward(self, cond_vec: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        """cond_vec: (T_a, cond_dim), x_norm: (Q,) in [0,1] -> confidence (T_a, Q)."""
        T_a = cond_vec.shape[0]
        Q = x_norm.shape[0]
        fourier = self.fourier_encode(x_norm)                        # (Q, 2F)
        fourier_b = fourier.unsqueeze(0).expand(T_a, -1, -1)          # (T_a, Q, 2F)
        cond_b = cond_vec.unsqueeze(1).expand(-1, Q, -1)               # (T_a, Q, cond_dim)
        feat = torch.cat([fourier_b, cond_b], dim=-1)                    # (T_a, Q, 2F+cond_dim)
        return self.mlp(feat).squeeze(-1)                                 # (T_a, Q)


class PathPredictorINR(nn.Module):
    def __init__(self, channels: int = 128, num_heads: int = 4,
                 cond_dim: int = 64, hidden_dim: int = 256,
                 fourier_freqs=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.axial = AxialAttentionBlock(channels, num_heads)
        self.cond_extractor = nn.Linear(channels, cond_dim)
        self.inr_head = INRHead(cond_dim, hidden_dim, fourier_freqs)

    def forward(self, sim: torch.Tensor, query_x_norm: torch.Tensor) -> torch.Tensor:
        """sim: (T_a, N_s), query_x_norm: (Q,) in [0,1] -> confidence (T_a, Q)."""
        x = sim.unsqueeze(0).unsqueeze(0)          # (1, 1, T_a, N_s)
        x = self.conv_stem(x)                       # (1, C, T_a, N_s)
        x = x.squeeze(0).permute(1, 2, 0)            # (T_a, N_s, C)
        x = self.axial(x)                            # (T_a, N_s, C)
        cond_per_frame = x.mean(dim=1)                # (T_a, C) -- mean over score axis
        cond_vec = self.cond_extractor(cond_per_frame)  # (T_a, cond_dim)
        return self.inr_head(cond_vec, query_x_norm)     # (T_a, Q)


class M06INRHead(nn.Module):
    def __init__(self, mert_dim: int = 768, dinov2_dim: int = 768,
                 hidden_dim: int = 512, embed_dim: int = 256,
                 path_channels: int = 128, attention_heads: int = 4,
                 cond_dim: int = 64, inr_hidden_dim: int = 256,
                 fourier_freqs=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.encoder = M04DenseTokens(mert_dim, dinov2_dim, hidden_dim, embed_dim)
        self.path_predictor = PathPredictorINR(
            path_channels, attention_heads, cond_dim, inr_hidden_dim, fourier_freqs)

    def forward(self, audio: torch.Tensor, score: torch.Tensor,
                query_x: torch.Tensor) -> dict:
        """audio: (T_a, 768), score: (N_cols, 16, 768).
        query_x: (Q,) continuous pixel positions to evaluate (any density).
        """
        enc = self.encoder(audio, score)
        x_min, x_max = query_x.min(), query_x.max()
        x_norm = (query_x - x_min) / (x_max - x_min).clamp_min(1e-6)
        confidence = self.path_predictor(enc['sim'], x_norm)          # (T_a, Q)
        p = F.softmax(confidence, dim=-1)
        pred_pos = (p * query_x.view(1, -1)).sum(dim=-1)                # (T_a,) pixel space
        return {'confidence': confidence, 'p': p, 'pred_pos': pred_pos, 'sim': enc['sim']}
