"""Variant A: dual-encoder + projection heads, joined by cosine similarity."""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import AudioEncoder, ImageEncoder


@dataclass
class AlignmentModelConfig:
    shared_dim: int = 256

    # audio
    audio_model_id: str = "m-a-p/MERT-v1-95M"
    pool_hz: int = 10
    freeze_audio: bool = True
    lora_rank_audio: int = 0   # 0 = no LoRA; >0 = LoRA rank (e.g. 8)

    # image
    image_model_id: str = "google/vit-base-patch16-224-in21k"
    tile_size: int = 224
    tile_stride: int = 56
    freeze_image: bool = True
    lora_rank_image: int = 0   # 0 = no LoRA; >0 = LoRA rank (e.g. 8)

    # projection heads
    head_dropout: float = 0.0


class _ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(self.norm(x)))


class AlignmentModel(nn.Module):
    """Variant A model.

    Forward inputs:
        audio:        (B, T_samples) float32 waveform at 24 kHz
        image:        (B, 3, 224, W) image strip (W can vary across batches)
        image_mask:   (B, W) bool, optional; True where the strip is real content

    Forward outputs (dict):
        sim:          (B, T_audio_pooled, N_tiles) cosine similarity matrix
        audio_embeds: (B, T_audio_pooled, shared_dim) L2-normalised
        image_embeds: (B, N_tiles, shared_dim) L2-normalised
        tile_mask:    (B, N_tiles) bool, True for valid tiles
    """

    def __init__(self, cfg: AlignmentModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or AlignmentModelConfig()

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
        self.audio_proj = _ProjectionHead(
            self.audio_enc.d_audio, self.cfg.shared_dim, self.cfg.head_dropout
        )
        self.image_proj = _ProjectionHead(
            self.image_enc.d_image, self.cfg.shared_dim, self.cfg.head_dropout
        )

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
        a_feats = self.audio_enc(audio)                              # (B, T_pool, d_a)
        i_feats, tile_mask = self.image_enc(image, image_mask)       # (B, N, d_i), (B, N)

        a = F.normalize(self.audio_proj(a_feats), dim=-1)            # (B, T_pool, d)
        i = F.normalize(self.image_proj(i_feats), dim=-1)            # (B, N, d)

        # cosine similarity matrix (since both are L2-normalised, dot product == cosine)
        sim = torch.einsum("btd,bnd->btn", a, i)                     # (B, T_pool, N)

        # mask out invalid tile columns (set similarity to a very small number)
        # so they don't dominate the soft-min in SoftDTW.
        if image_mask is not None:
            sim = sim.masked_fill(~tile_mask[:, None, :], -1.0)

        return {
            "sim":          sim,
            "audio_embeds": a,
            "image_embeds": i,
            "tile_mask":    tile_mask,
        }
