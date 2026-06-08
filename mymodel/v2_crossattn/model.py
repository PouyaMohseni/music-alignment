"""v2 — Cross-Attention Fusion model.

Architecture vs v1:
  v1: audio_embed → proj → L2-norm
      image_embed → proj → L2-norm
      sim = dot(audio, image)        # independent embeddings, late fusion

  v2: audio_embed → audio_proj → Q_a   (B, T, d)
      image_embed → image_proj → K_i   (B, N, d)
      audio attends to image: A_a = CrossAttn(Q=Q_a, K=K_i, V=K_i)  (B, T, d)
      image attends to audio: A_i = CrossAttn(Q=K_i, K=Q_a, V=Q_a)  (B, N, d)
      sim = dot(L2(A_a), L2(A_i))    # context-aware similarity

Cross-attention lets each modality see the other before the similarity is
computed, making the similarity matrix much more discriminative for alignment.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse encoders from v1 — identical backbone/LoRA setup
from ..v1_baseline.encoders import AudioEncoder, ImageEncoder


@dataclass
class CrossAttnModelConfig:
    shared_dim: int = 256
    n_heads: int = 4
    attn_dropout: float = 0.1
    head_dropout: float = 0.1

    audio_model_id: str = "m-a-p/MERT-v1-95M"
    pool_hz: int = 10
    freeze_audio: bool = True
    lora_rank_audio: int = 4

    image_model_id: str = "google/vit-base-patch16-224-in21k"
    tile_size: int = 224
    tile_stride: int = 56
    freeze_image: bool = True
    lora_rank_image: int = 4


class _ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(self.norm(x)))


class _CrossAttention(nn.Module):
    """Single cross-attention layer: Q from one modality, K/V from the other."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,   # (B, T_q, d)
        context: torch.Tensor, # (B, T_kv, d)
    ) -> torch.Tensor:
        # cross-attention + residual
        attn_out, _ = self.attn(query, context, context)
        x = self.norm(query + attn_out)
        # feed-forward + residual
        x = self.norm2(x + self.ff(x))
        return x


class CrossAttnAlignmentModel(nn.Module):
    """v2 alignment model with cross-attention fusion.

    Forward inputs:
        audio:        (B, T_samples) float32 waveform at 24 kHz
        image:        (B, 3, 224, W) image strip
        image_mask:   (B, W) bool, optional

    Forward outputs (dict):
        sim:          (B, T_audio_pooled, N_tiles) cosine similarity
        audio_embeds: (B, T_audio_pooled, d) after cross-attention
        image_embeds: (B, N_tiles, d) after cross-attention
        tile_mask:    (B, N_tiles) bool
    """

    def __init__(self, cfg: CrossAttnModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or CrossAttnModelConfig()

        self.audio_enc = AudioEncoder(
            model_id=self.cfg.audio_model_id,
            pool_hz=self.cfg.pool_hz,
            freeze=self.cfg.freeze_audio,
            lora_rank=self.cfg.lora_rank_audio,
        )
        self.image_enc = ImageEncoder(
            model_id=self.cfg.image_model_id,
            tile_size=self.cfg.tile_size,
            stride=self.cfg.tile_stride,
            freeze=self.cfg.freeze_image,
            lora_rank=self.cfg.lora_rank_image,
        )
        d = self.cfg.shared_dim
        self.audio_proj = _ProjectionHead(
            self.audio_enc.d_audio, d, self.cfg.head_dropout
        )
        self.image_proj = _ProjectionHead(
            self.image_enc.d_image, d, self.cfg.head_dropout
        )
        self.audio_cross = _CrossAttention(d, self.cfg.n_heads, self.cfg.attn_dropout)
        self.image_cross = _CrossAttention(d, self.cfg.n_heads, self.cfg.attn_dropout)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(
        self,
        audio: torch.Tensor,
        image: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> dict:
        a_feats = self.audio_enc(audio)                              # (B, T, d_a)
        i_feats, tile_mask = self.image_enc(image, image_mask)       # (B, N, d_i)

        a = self.audio_proj(a_feats)                                 # (B, T, d)
        i = self.image_proj(i_feats)                                 # (B, N, d)

        # cross-attention: each modality attends to the other
        a = self.audio_cross(query=a, context=i)                     # (B, T, d)
        i = self.image_cross(query=i, context=a)                     # (B, N, d)

        a = F.normalize(a, dim=-1)
        i = F.normalize(i, dim=-1)

        sim = torch.einsum("btd,bnd->btn", a, i)                     # (B, T, N)

        if image_mask is not None:
            sim = sim.masked_fill(~tile_mask[:, None, :], -1.0)

        return {
            "sim":          sim,
            "audio_embeds": a,
            "image_embeds": i,
            "tile_mask":    tile_mask,
        }
