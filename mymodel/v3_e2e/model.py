"""Variant C — End-to-End alignment model.

Both encoders run live (LoRA adapters trainable). The full score strip passes
through ViT; a 5-second audio window passes through MERT. A cross-attention head
maps both into a shared space and produces a (T_window, N_strip) similarity
matrix consumed by the distance-aware localization loss.

Key differences from v3_fullseq:
- Encoders are IN the graph (LoRA unfrozen)
- No precomputed embeddings — everything runs at train time
- Audio is windowed (5 s); score strip is always full
- Two-group optimizer: low LR for encoder LoRA, higher for head
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..v1_baseline.encoders import AudioEncoder, ImageEncoder


@dataclass
class E2EModelConfig:
    # encoders
    audio_model_id: str = "m-a-p/MERT-v1-95M"
    pool_hz: int = 10
    lora_rank_audio: int = 4

    image_model_id: str = "google/vit-base-patch16-224-in21k"
    tile_size: int = 224
    tile_stride: int = 56
    lora_rank_image: int = 4

    # head
    shared_dim: int = 256
    n_heads: int = 4
    n_cross_layers: int = 1
    dropout: float = 0.1


class _ProjectionHead(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.proj(self.drop(self.norm(x)))


class _CrossAttnLayer(nn.Module):
    def __init__(self, d, n_heads, dropout):
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


class E2EAlignmentModel(nn.Module):
    """End-to-end model with live LoRA encoders.

    forward(audio_window, strip_image) -> dict with:
        sim          (B, T_window, N_strip) cosine similarity
        audio_embeds (B, T_window, d)
        image_embeds (B, N_strip, d)

    audio_window : (B, T_samples) at 24 kHz
    strip_image  : (B, 3, 224, W) full-strip image
    """

    def __init__(self, cfg: E2EModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or E2EModelConfig()

        # LoRA adapters trainable; base weights frozen
        self.audio_enc = AudioEncoder(
            model_id=self.cfg.audio_model_id,
            pool_hz=self.cfg.pool_hz,
            freeze=True,
            lora_rank=self.cfg.lora_rank_audio,
        )
        self.image_enc = ImageEncoder(
            model_id=self.cfg.image_model_id,
            tile_size=self.cfg.tile_size,
            stride=self.cfg.tile_stride,
            freeze=True,
            lora_rank=self.cfg.lora_rank_image,
        )

        d = self.cfg.shared_dim
        self.audio_proj = _ProjectionHead(self.audio_enc.d_audio, d, self.cfg.dropout)
        self.image_proj = _ProjectionHead(self.image_enc.d_image, d, self.cfg.dropout)
        self.audio_cross = nn.ModuleList(
            [_CrossAttnLayer(d, self.cfg.n_heads, self.cfg.dropout)
             for _ in range(self.cfg.n_cross_layers)])
        self.image_cross = nn.ModuleList(
            [_CrossAttnLayer(d, self.cfg.n_heads, self.cfg.dropout)
             for _ in range(self.cfg.n_cross_layers)])

    def encoder_parameters(self):
        """LoRA adapter parameters only — low-LR group."""
        for enc in (self.audio_enc, self.image_enc):
            for p in enc.parameters():
                if p.requires_grad:
                    yield p

    def head_parameters(self):
        """Projection + cross-attention — high-LR group."""
        for mod in (self.audio_proj, self.image_proj,
                    self.audio_cross, self.image_cross):
            yield from mod.parameters()

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, audio_window: torch.Tensor,
                strip_image: torch.Tensor) -> dict:
        a_feat = self.audio_enc(audio_window)                     # (B, T, d_a)
        i_feat, tile_mask = self.image_enc(strip_image)           # (B, N, d_i)

        a = self.audio_proj(a_feat)                               # (B, T, d)
        i = self.image_proj(i_feat)                               # (B, N, d)

        for la, li in zip(self.audio_cross, self.image_cross):
            a_new = la(query=a, context=i)
            i_new = li(query=i, context=a)
            a, i = a_new, i_new

        a = F.normalize(a, dim=-1)
        i = F.normalize(i, dim=-1)
        sim = torch.einsum("btd,bnd->btn", a, i)                  # (B, T, N)

        return {"sim": sim, "audio_embeds": a, "image_embeds": i,
                "tile_mask": tile_mask}
