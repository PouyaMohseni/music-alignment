"""CADP M04 — Dense Tokens (no pooling).

Audio:  pre-computed MERT (T_a, 768) at 20fps — used densely, no chunk pooling.
Score:  pre-computed DINOv2 (N_cols, 16, 768) per piece. The 16 patch tokens
        per column are a 4x4 (height x width) grid; we mean-pool over height
        only, keeping 4 horizontal sub-positions per column for finer spatial
        resolution than a single column-center point.

Architecture:
  audio_emb = L2Norm(TwoLayerMLP(768->512->256)(audio))          # (T_a, E)
  score_vpool = mean(reshape(score, (N_cols,4,4,768)), dim=1)    # (N_cols,4,768)
  score_emb = L2Norm(TwoLayerMLP(768->512->256)(score_vpool))    # (N_cols*4, E)
  sim = audio_emb @ score_emb.T                                  # (T_a, N_cols*4)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class M04DenseTokens(nn.Module):
    def __init__(self, mert_dim: int = 768, dinov2_dim: int = 768,
                 hidden_dim: int = 512, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.audio_proj = TwoLayerMLP(mert_dim, hidden_dim, embed_dim)
        self.score_proj = TwoLayerMLP(dinov2_dim, hidden_dim, embed_dim)

    def encode_score(self, score: torch.Tensor) -> torch.Tensor:
        """score: (N_cols, 16, 768) -> score_emb (N_cols*4, E)."""
        N_cols = score.shape[0]
        score_grid = score.view(N_cols, 4, 4, -1)          # (N_cols, h=4, w=4, 768)
        score_vpool = score_grid.mean(dim=1)                 # (N_cols, w=4, 768)
        score_flat = score_vpool.reshape(N_cols * 4, -1)     # (N_cols*4, 768)
        return F.normalize(self.score_proj(score_flat), dim=-1)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (T_a, 768) -> audio_emb (T_a, E)."""
        return F.normalize(self.audio_proj(audio), dim=-1)

    def forward(self, audio: torch.Tensor, score: torch.Tensor) -> dict:
        """audio: (T_a, 768), score: (N_cols, 16, 768). No batch dim (single piece)."""
        audio_emb = self.encode_audio(audio)          # (T_a, E)
        score_emb = self.encode_score(score)           # (N_s=N_cols*4, E)
        sim = audio_emb @ score_emb.T                   # (T_a, N_s)
        return {'sim': sim, 'audio_emb': audio_emb, 'score_emb': score_emb}


def subcol_positions(n_cols: int, col_stride: float, col_w: float) -> torch.Tensor:
    """Pixel-space center of each of the N_cols*4 horizontal sub-positions.
    Column i (width col_w, stride col_stride) is split into 4 equal sub-bins;
    sub-position j in [0,4) is centered at i*col_stride + (j+0.5)*col_w/4.
    """
    i = torch.arange(n_cols).view(-1, 1).float()
    j = torch.arange(4).view(1, -1).float()
    pos = i * col_stride + (j + 0.5) * (col_w / 4.0)   # (N_cols, 4)
    return pos.reshape(-1)                              # (N_cols*4,)
