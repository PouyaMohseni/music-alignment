"""CADP M03 — LSTM Temporal Follower (v5_recurrent replica).

Audio:  pre-computed MERT (T_a, 768) at 20fps
Score:  pre-computed DINOv2 (N_cols, 16, 768) per piece

Architecture (same encoders/projections as M01, plus a BiLSTM over audio):
  1. Pool audio to N_chunks via mean (T_a → N_chunks)
  2. Linear(768, embed_dim) + L2-normalize → audio_emb (N_chunks, embed_dim)
  3. BiLSTM(embed_dim, lstm_hidden, lstm_layers, bidirectional) over audio_emb
  4. Linear(2*lstm_hidden, embed_dim) → audio_query, L2-normalize
  5. Mean-pool DINOv2 patches → Linear(768, embed_dim) + L2-norm → score_emb
  6. sim_matrix = audio_query @ score_emb.T → (N_chunks, N_cols)

Trainable: audio_proj, score_proj, lstm, query_proj.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class M03LSTMTemporal(nn.Module):
    def __init__(self, mert_dim: int = 768, dinov2_dim: int = 768,
                 embed_dim: int = 256, n_audio_chunks: int = 20,
                 lstm_hidden: int = 512, lstm_layers: int = 2,
                 lstm_bidirectional: bool = True):
        super().__init__()
        self.n_audio_chunks = n_audio_chunks
        self.audio_proj = nn.Linear(mert_dim, embed_dim)
        self.score_proj = nn.Linear(dinov2_dim, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, bidirectional=lstm_bidirectional,
            batch_first=True)
        lstm_out_dim = lstm_hidden * (2 if lstm_bidirectional else 1)
        self.query_proj = nn.Linear(lstm_out_dim, embed_dim)

    def forward(self,
                audio: torch.Tensor,   # (B, T_a, 768)
                score: torch.Tensor,   # (N_cols, 16, 768) or (B, N_cols, 16, 768)
                hidden=None,
                n_chunks: int | None = None,
                ) -> dict:
        B, T_a, _ = audio.shape
        # n_chunks lets eval pool at the SAME per-chunk time resolution seen in
        # training (win_sec/n_audio_chunks) instead of collapsing a whole piece
        # to a fixed count of chunks, which coarsens temporal resolution ~14x.
        K = n_chunks if n_chunks is not None else self.n_audio_chunks

        # Pool audio: T_a → K chunks
        audio_t = audio.permute(0, 2, 1)                                   # (B, 768, T_a)
        audio_pooled = F.adaptive_avg_pool1d(audio_t, K).permute(0, 2, 1)  # (B, K, 768)
        audio_emb = self.audio_proj(audio_pooled)                          # (B, K, E) — no norm before LSTM

        lstm_out, hidden = self.lstm(audio_emb, hidden)     # (B, K, lstm_out_dim)
        audio_query = F.normalize(self.query_proj(lstm_out), dim=-1)  # (B, K, E)

        # Pool score patches: (N_cols, 16, 768) → (N_cols, 768)
        if score.dim() == 3:
            score_pooled = score.mean(dim=1)       # (N_cols, 768)
        else:
            score_pooled = score.mean(dim=2)       # (B, N_cols, 768)
        score_emb = F.normalize(self.score_proj(score_pooled), dim=-1)   # (N_cols, E)

        if score_emb.dim() == 2:
            sim = torch.einsum('bkd,nd->bkn', audio_query, score_emb)
        else:
            sim = torch.bmm(audio_query, score_emb.transpose(1, 2))

        return {'sim': sim, 'audio_emb': audio_query, 'score_emb': score_emb, 'hidden': hidden}
