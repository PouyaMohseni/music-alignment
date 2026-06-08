"""v3 full-sequence alignment head.

Operates on PRECOMPUTED frozen encoder embeddings (see precompute.py), so the
foundation models are not in the graph at all. Trainable parts:
  - audio projection head   (Da -> d)
  - image projection head   (Di -> d)
  - optional cross-attention layers (audio <-> score over the full sequence)

Produces a (T, N) similarity matrix consumed by expected_distance_loss.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FullSeqModelConfig:
    d_audio: int = 768
    d_image: int = 768
    shared_dim: int = 256
    n_heads: int = 4
    n_cross_layers: int = 1     # 0 disables cross-attention (pure projection)
    dropout: float = 0.1


class _ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.proj(self.drop(self.norm(x)))


class _CrossAttnLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, query, context):
        a, _ = self.attn(query, context, context)
        x = self.norm(query + a)
        return self.norm2(x + self.ff(x))


class FullSeqAlignmentModel(nn.Module):
    """Lightweight head over cached embeddings.

    forward(audio_emb, tile_emb) -> dict with:
        sim          (B, T, N) cosine similarity
        audio_embeds (B, T, d)
        image_embeds (B, N, d)
    Inputs are (B, T, Da) / (B, N, Di); B is typically 1 (one full piece).
    """

    def __init__(self, cfg: FullSeqModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or FullSeqModelConfig()
        d = self.cfg.shared_dim
        self.audio_proj = _ProjectionHead(self.cfg.d_audio, d, self.cfg.dropout)
        self.image_proj = _ProjectionHead(self.cfg.d_image, d, self.cfg.dropout)
        self.audio_layers = nn.ModuleList(
            [_CrossAttnLayer(d, self.cfg.n_heads, self.cfg.dropout)
             for _ in range(self.cfg.n_cross_layers)])
        self.image_layers = nn.ModuleList(
            [_CrossAttnLayer(d, self.cfg.n_heads, self.cfg.dropout)
             for _ in range(self.cfg.n_cross_layers)])

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, audio_emb: torch.Tensor, tile_emb: torch.Tensor) -> dict:
        a = self.audio_proj(audio_emb)            # (B, T, d)
        i = self.image_proj(tile_emb)             # (B, N, d)
        for la, li in zip(self.audio_layers, self.image_layers):
            a_new = la(query=a, context=i)
            i_new = li(query=i, context=a)
            a, i = a_new, i_new
        a = F.normalize(a, dim=-1)
        i = F.normalize(i, dim=-1)
        sim = torch.einsum("btd,bnd->btn", a, i)
        return {"sim": sim, "audio_embeds": a, "image_embeds": i}
