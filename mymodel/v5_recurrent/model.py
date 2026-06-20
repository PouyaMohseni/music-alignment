"""v5 — Recurrent score follower with LSTM temporal conditioning.

Same frozen-embedding pipeline as v3 (proj + cross-attn), then an LSTM reads
the projected audio sequence and produces a temporally-conditioned query at
each frame.  The query is matched against all score tiles.

Training loss  : cross-entropy(logits[t], nearest_tile[t]) per valid frame.
Inference      : DTW on the (T, N) logit matrix (globally optimal path).

Key difference vs v3/v4: the model conditions each frame's prediction on
where it was before (LSTM state), addressing the memoryless-retrieval failure
mode (RC1).

Config flags:
  lstm_bidirectional : bidirectional LSTM — doubles output dim, uses full sequence context
  residual           : logits = LSTM_logits + raw_sim (LSTM learns a correction on top of v3)
  pitch_hidden       : >0 adds 88-key pitch heads for aux BCE supervision at training time
                       (gradient-only — not fused into logits at inference)
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PITCH = 88


@dataclass
class RecurrentConfig:
    d_audio: int = 768
    d_image: int = 768
    shared_dim: int = 256
    n_heads: int = 4
    n_cross_layers: int = 1
    dropout: float = 0.1
    lstm_hidden: int = 256
    lstm_layers: int = 1
    lstm_bidirectional: bool = False   # bidir LSTM — 2x hidden for query_proj input
    residual: bool = False             # logits = LSTM_logits + raw_sim
    pitch_hidden: int = 0              # 0 = no pitch heads; >0 = add aux pitch supervision
    pitch_on_aligned: bool = False     # E0: tap ALIGNED feature a/i (shapes the matched
                                       # rep) vs raw frozen emb (the inert v4c/v5k wiring)


class _PitchHead(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, N_PITCH))

    def forward(self, x):
        return self.net(x)


class _ProjHead(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.proj(self.drop(self.norm(x)))


class _CrossAttn(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, q, ctx):
        a, _ = self.attn(q, ctx, ctx)
        x = self.norm(q + a)
        return self.norm2(x + self.ff(x))


class RecurrentFollower(nn.Module):
    """forward(audio_emb (B,T,Da), tile_emb (B,N,Di)) -> dict:
        logits  (B, T, N)  — LSTM-conditioned logits (used for DTW decode)
        sim     (B, T, N)  — raw cosine similarity (retrieval metrics)
    """

    def __init__(self, cfg: RecurrentConfig | None = None):
        super().__init__()
        self.cfg = cfg or RecurrentConfig()
        d = self.cfg.shared_dim
        self.audio_proj = _ProjHead(self.cfg.d_audio, d, self.cfg.dropout)
        self.image_proj = _ProjHead(self.cfg.d_image, d, self.cfg.dropout)
        self.audio_layers = nn.ModuleList(
            [_CrossAttn(d, self.cfg.n_heads, self.cfg.dropout)
             for _ in range(self.cfg.n_cross_layers)])
        self.image_layers = nn.ModuleList(
            [_CrossAttn(d, self.cfg.n_heads, self.cfg.dropout)
             for _ in range(self.cfg.n_cross_layers)])
        self.lstm = nn.LSTM(d, self.cfg.lstm_hidden, self.cfg.lstm_layers,
                            batch_first=True,
                            bidirectional=self.cfg.lstm_bidirectional)
        lstm_out_dim = self.cfg.lstm_hidden * (2 if self.cfg.lstm_bidirectional else 1)
        self.query_proj = nn.Linear(lstm_out_dim, d)
        if self.cfg.pitch_hidden > 0:
            # E0 fix: when pitch_on_aligned, the pitch head consumes the ALIGNED feature
            # (dim = shared_dim) so its gradient flows back through audio_proj/image_proj
            # and the cross-attention — i.e. it reshapes the representation that is
            # actually matched. Legacy (False) reads the raw *frozen* embedding, so the
            # gradient dead-ends and shapes nothing (the v4c/v5k wiring bug, §9.1).
            pd_a = d if self.cfg.pitch_on_aligned else self.cfg.d_audio
            pd_s = d if self.cfg.pitch_on_aligned else self.cfg.d_image
            self.audio_pitch = _PitchHead(pd_a, self.cfg.pitch_hidden)
            self.score_pitch = _PitchHead(pd_s, self.cfg.pitch_hidden)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, audio_emb, tile_emb):
        a = self.audio_proj(audio_emb)    # (B, T, d)
        i = self.image_proj(tile_emb)     # (B, N, d)
        for la, li in zip(self.audio_layers, self.image_layers):
            a, i = la(a, i), li(i, a)

        i_n = F.normalize(i, dim=-1)
        sim = torch.einsum("btd,bnd->btn", F.normalize(a, dim=-1), i_n)

        lstm_out, _ = self.lstm(a)
        q = F.normalize(self.query_proj(lstm_out), dim=-1)
        lstm_logits = torch.einsum("btd,bnd->btn", q, i_n)

        # residual: LSTM corrects on top of raw cosine sim instead of replacing it
        logits = sim + lstm_logits if self.cfg.residual else lstm_logits

        out = {"logits": logits, "sim": sim}
        if self.cfg.pitch_hidden > 0:
            if self.cfg.pitch_on_aligned:
                # tap aligned features (post proj + cross-attn) — the exact tensors that
                # produce `sim` and feed the LSTM. Pitch BCE now shapes the matched rep.
                out["audio_pitch_logits"] = self.audio_pitch(a)       # (B,T,88)
                out["score_pitch_logits"] = self.score_pitch(i)       # (B,N,88)
            else:
                out["audio_pitch_logits"] = self.audio_pitch(audio_emb)   # legacy raw frozen
                out["score_pitch_logits"] = self.score_pitch(tile_emb)
        return out
