"""CADP M05 — Learned Path Predictor.

Replaces DTW-based decoding (M01/M03's failure mode: a single hard, monotonic
whole-piece path that cascades into catastrophic misalignment whenever tempo
drift or a musical repeat pushes the true alignment outside the DTW band —
see REDESIGN notes on Chopin/Satie outliers) with a learned, per-frame
predictor. Every audio frame's score position is read off independently via
attention-refined logits + soft-argmax — there is no global monotonic
constraint to break, so one bad local match can't derail the rest of the
piece the way a global DTW backtrack can.

Architecture (operates on M04's dense sim matrix, treated as a 1-channel
"image" of shape (T_a, N_s)):
  conv_stem: 3x Conv2d+ReLU, channels 1->32->64->128
  axial attention along the score axis (each audio frame attends over its
    own row of score positions), then along the audio axis (each score
    position attends over its own column of audio frames)
  head: per-cell Linear(128,1) -> logits (T_a, N_s)
  predicted position per audio frame: soft-argmax of logits over the score
    axis, weighted against the actual pixel positions of each score bin.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mymodel.cadp.m04_model import M04DenseTokens


class AxialAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn_score = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.attn_audio = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm_score = nn.LayerNorm(channels)
        self.norm_audio = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T_a, N_s, C)."""
        T_a, N_s, C = x.shape

        # Attend along the score axis: each audio frame (batch elem) attends
        # over its own N_s score positions.
        out, _ = self.attn_score(x, x, x)                       # (T_a, N_s, C)
        x = self.norm_score(x + out)

        # Attend along the audio axis: each score position (batch elem)
        # attends over its own T_a audio frames.
        x_t = x.transpose(0, 1)                                  # (N_s, T_a, C)
        out, _ = self.attn_audio(x_t, x_t, x_t)                  # (N_s, T_a, C)
        x_t = self.norm_audio(x_t + out)
        return x_t.transpose(0, 1)                                # (T_a, N_s, C)


class PathPredictor(nn.Module):
    def __init__(self, channels: int = 128, num_heads: int = 4):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.axial = AxialAttentionBlock(channels, num_heads)
        self.head = nn.Linear(channels, 1)

    def forward(self, sim: torch.Tensor) -> torch.Tensor:
        """sim: (T_a, N_s) -> logits (T_a, N_s)."""
        x = sim.unsqueeze(0).unsqueeze(0)          # (1, 1, T_a, N_s)
        x = self.conv_stem(x)                       # (1, C, T_a, N_s)
        x = x.squeeze(0).permute(1, 2, 0)            # (T_a, N_s, C)
        x = self.axial(x)                            # (T_a, N_s, C)
        logits = self.head(x).squeeze(-1)             # (T_a, N_s)
        return logits


class M05LearnedPathPredictor(nn.Module):
    def __init__(self, mert_dim: int = 768, dinov2_dim: int = 768,
                 hidden_dim: int = 512, embed_dim: int = 256,
                 path_channels: int = 128, attention_heads: int = 4):
        super().__init__()
        self.encoder = M04DenseTokens(mert_dim, dinov2_dim, hidden_dim, embed_dim)
        self.path_predictor = PathPredictor(path_channels, attention_heads)

    def forward(self, audio: torch.Tensor, score: torch.Tensor,
                pos_subcol: torch.Tensor, temperature: float = 0.5) -> dict:
        """audio: (T_a, 768), score: (N_cols, 16, 768).
        pos_subcol: (N_s,) pixel position of each score sub-bin (N_s = N_cols*4).
        Returns per-frame soft-argmax predicted pixel position (T_a,).
        """
        enc = self.encoder(audio, score)
        logits = self.path_predictor(enc['sim'])            # (T_a, N_s)
        p = F.softmax(logits / temperature, dim=-1)           # (T_a, N_s)
        pred_pos = (p * pos_subcol.view(1, -1)).sum(dim=-1)    # (T_a,) pixel space
        return {'logits': logits, 'p': p, 'pred_pos': pred_pos, 'sim': enc['sim']}
