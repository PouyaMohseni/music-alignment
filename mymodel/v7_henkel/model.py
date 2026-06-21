"""v7 — Henkel-style FiLM-conditioned score follower.

Core idea from Henkel, Kelz & Widmer (ISMIR 2020): an LSTM over audio history
produces per-frame FiLM parameters (γ, β) that modulate the projected score tile
representation before matching.  This gives the model a dynamic, state-conditioned
score view — directly addressing the tile-bucket-DTW framing wall (RC1 from E1).

Adapted to our frozen-embedding pipeline (no pianoroll at inference):
  audio : MERT last-hidden-state (precomputed, frozen)
  score : ViT tile embeddings   (precomputed, frozen)

Forward interface identical to v5 RecurrentFollower:
  (audio_emb (B,T,Da), tile_emb (B,N,Di)) → {"logits": (B,T,N), "sim": (B,T,N)}
  → drop-in replacement for v5 train.py / eval.py.

FiLM dot-product avoids materialising (B,T,N,d) by decomposing:
  logits[b,t,n] = (q[b,t] ⊙ (1+γ[b,t])) · i[b,n]
                + (q[b,t] ⊙  β[b,t]).sum()
which is O(BTdN) multiply-accumulate, same as standard matmul.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HenkelConfig:
    d_audio: int = 768
    d_image: int = 768
    shared_dim: int = 256
    n_heads: int = 4
    n_cross_layers: int = 1
    dropout: float = 0.1
    lstm_hidden: int = 512
    lstm_layers: int = 1
    film_hidden: int = 256    # hidden dim of the FiLM parameter generator MLP
    residual: bool = True     # logits = FiLM_logits + raw_sim  (safe warm-start anchor)


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
        self.norm  = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, q, ctx):
        a, _ = self.attn(q, ctx, ctx)
        x = self.norm(q + a)
        return self.norm2(x + self.ff(x))


class HenkelFollower(nn.Module):
    """FiLM-conditioned score follower using frozen MERT + ViT embeddings.

    Steps per forward pass:
      1. Project audio + score to shared_dim.
      2. Mutual cross-attention (same as v5).
      3. Unidirectional LSTM over projected audio → context h  (B, T, lstm_h).
      4. FiLM generator: h → (γ, β), each (B, T, d).
      5. Conditioned query: q = query_proj(h)                  (B, T, d).
      6. FiLM dot-product:
           film_logits = (q ⊙ (1+γ)) @ i.T  +  (q ⊙ β).sum(-1, keepdim=True)
      7. logits = raw_sim + film_logits  (residual=True keeps a v5-like anchor).
    """

    def __init__(self, cfg: HenkelConfig | None = None):
        super().__init__()
        self.cfg = cfg or HenkelConfig()
        d  = self.cfg.shared_dim
        lh = self.cfg.lstm_hidden

        self.audio_proj = _ProjHead(self.cfg.d_audio, d, self.cfg.dropout)
        self.image_proj = _ProjHead(self.cfg.d_image, d, self.cfg.dropout)

        self.audio_layers = nn.ModuleList(
            [_CrossAttn(d, self.cfg.n_heads, self.cfg.dropout)
             for _ in range(self.cfg.n_cross_layers)])
        self.image_layers = nn.ModuleList(
            [_CrossAttn(d, self.cfg.n_heads, self.cfg.dropout)
             for _ in range(self.cfg.n_cross_layers)])

        # Unidirectional: causal history, consistent with online following.
        self.lstm = nn.LSTM(d, lh, self.cfg.lstm_layers, batch_first=True)

        self.film_gen = nn.Sequential(
            nn.LayerNorm(lh),
            nn.Linear(lh, self.cfg.film_hidden),
            nn.GELU(),
            nn.Linear(self.cfg.film_hidden, 2 * d))  # → (γ, β) concatenated

        self.query_proj = nn.Linear(lh, d)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, audio_emb, tile_emb):
        """
        audio_emb : (B, T, d_audio)
        tile_emb  : (B, N, d_image)
        returns   : {"logits": (B,T,N), "sim": (B,T,N)}
        """
        a = self.audio_proj(audio_emb)    # (B, T, d)
        i = self.image_proj(tile_emb)     # (B, N, d)

        for la, li in zip(self.audio_layers, self.image_layers):
            a, i = la(a, i), li(i, a)

        # Raw cosine similarity (retrieval metrics + residual anchor)
        sim = torch.einsum("btd,bnd->btn",
                           F.normalize(a, dim=-1),
                           F.normalize(i, dim=-1))

        h, _ = self.lstm(a)               # (B, T, lh)

        film   = self.film_gen(h)         # (B, T, 2d)
        gamma, beta = film.chunk(2, dim=-1)   # (B, T, d) each

        q = self.query_proj(h)            # (B, T, d)

        # FiLM dot-product — no (B,T,N,d) materialisation needed:
        #   logits[b,t,n] = (q⊙(1+γ))·i[n]  +  (q⊙β).sum()
        q_scaled    = q * (1.0 + gamma)                       # (B, T, d)
        film_logits = torch.bmm(q_scaled, i.permute(0, 2, 1)) # (B, T, N)
        film_logits = film_logits + (q * beta).sum(-1, keepdim=True)  # bias term

        logits = (sim + film_logits) if self.cfg.residual else film_logits

        return {"logits": logits, "sim": sim}
