"""CADP M01 — Frozen Baseline.

Audio:  pre-computed MERT (T_a, 768) at 20fps
Score:  pre-computed DINOv2 (N_cols, 16, 768) per piece

Architecture:
  1. Pool audio to N_chunks via mean (T_a → N_chunks)
  2. Linear(768, embed_dim) + L2-normalize → audio_emb (N_chunks, embed_dim)
  3. Mean-pool DINOv2 patches → Linear(768, embed_dim) + L2-norm → score_emb
  4. sim_matrix = audio_emb @ score_emb.T  → (N_chunks, N_cols)

Trainable: audio_proj, score_proj only (~400K params).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class M01FrozenBaseline(nn.Module):
    def __init__(self, mert_dim: int = 768, dinov2_dim: int = 768,
                 embed_dim: int = 256, n_audio_chunks: int = 20):
        super().__init__()
        self.n_audio_chunks = n_audio_chunks
        self.audio_proj = nn.Linear(mert_dim, embed_dim)
        self.score_proj = nn.Linear(dinov2_dim, embed_dim)

    def forward(self,
                audio: torch.Tensor,   # (B, T_a, 768)
                score: torch.Tensor,   # (N_cols, 16, 768) or (B, N_cols, 16, 768)
                n_chunks: int | None = None,
                ) -> dict:
        B, T_a, _ = audio.shape
        # n_chunks lets eval pool at the SAME per-chunk time resolution seen in
        # training (win_sec/n_audio_chunks) instead of collapsing a whole piece
        # to a fixed count of chunks, which coarsens temporal resolution ~14x.
        K = n_chunks if n_chunks is not None else self.n_audio_chunks

        # Pool audio: T_a → K chunks
        # Use adaptive_avg_pool1d for clean pooling
        # audio: (B, T_a, 768) → (B, 768, T_a) → pool → (B, 768, K) → (B, K, 768)
        audio_t = audio.permute(0, 2, 1)                            # (B, 768, T_a)
        audio_pooled = F.adaptive_avg_pool1d(audio_t, K).permute(0, 2, 1)  # (B, K, 768)
        audio_emb = F.normalize(self.audio_proj(audio_pooled), dim=-1)     # (B, K, E)

        # Pool score patches: (N_cols, 16, 768) → (N_cols, 768)
        if score.dim() == 3:
            score_pooled = score.mean(dim=1)       # (N_cols, 768)
        else:
            score_pooled = score.mean(dim=2)       # (B, N_cols, 768)
        score_emb = F.normalize(self.score_proj(score_pooled), dim=-1)   # (N_cols, E)

        if score_emb.dim() == 2:
            # sim: (B, K, N_cols)
            sim = torch.einsum('bkd,nd->bkn', audio_emb, score_emb)
        else:
            sim = torch.bmm(audio_emb, score_emb.transpose(1, 2))

        return {'sim': sim, 'audio_emb': audio_emb, 'score_emb': score_emb}
