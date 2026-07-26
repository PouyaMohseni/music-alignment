"""Dinov2ScoreTower -- frozen DINOv2-base (facebook/dinov2-base) as an
alternative to D1's from-scratch ScoreTower and the failed MuSViT attempt,
matching the SAME output interface: strip -> (W_col, d_model) L2-normalized
per-column embeddings.

Why DINOv2 over MuSViT: MuSViT's frozen features showed near-zero distance
correlation across strip position (see musvit_tower.py) and its overfit test
never reached the bar (0.65 vs 0.90 target) -- a real negative result, not
just an integration gap. DINOv2 is a DIFFERENT, PROVEN backbone in this exact
project: its native-page tiling scheme already powers two currently-running
experiments (extensions/hooks/dinov2_full_encoder_patch.py's
ConditionalUNetDINOv2Visual, mert-dinov2-crossattn) with a passing smoke test
and real training in progress -- i.e. its features are already known to
support real cross-attention/decoder use, unlike MuSViT which had never been
validated end-to-end before this session.

One honest caveat: the EXISTING precomputed DINOv2 tiles
(/scratch/pmohseni/dinov2_emb_tiled_native/) are keyed by NATIVE PAGE, using
one CLS token per whole 224x224 tile -- coarse (an MSMD page tiles to only
~13-20 tiles total) and, more importantly, in NATIVE-PAGE pixel space, which
D1/M1's STRIP format (a horizontal concatenation of staff-system crops) has
no clean provenance back to without re-deriving the strip-building pipeline's
own page/staff bookkeeping. Rather than solve that remapping problem, this
tower runs DINOv2 FRESH on the STRIP itself (frozen, no training cost beyond
inference), reusing the same tiling MECHANICS as musvit_tower.py's
strip_to_tiles (native-height tiles slid along width, upsampled if the D1
strip's scale_factor-reduced height is below tile size) -- but taking the
FULL per-tile PATCH GRID (16x16 for DINOv2's patch_size=14 at a 224x224 tile),
not just the tile's CLS token, since CLS-per-tile would only give ~13-20
columns per piece, far too coarse for onset-level resolution (M1 needs up to
~90 onset columns per piece).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mymodel.m1_monotonic.musvit_tower import strip_to_tiles, _sinusoidal_pe

_TILE = 224
_PATCH = 14
_GRID = _TILE // _PATCH   # 16


class Dinov2ScoreTower(nn.Module):
    """Interface-compatible with mymodel.d1_align_matrix.model.ScoreTower and
    mymodel.m1_monotonic.musvit_tower.MuSViTScoreTower: forward(strip) ->
    (W_col, d_model) L2-normalized per-column embeddings."""

    def __init__(self, d_model: int = 128, n_ctx_layers: int = 2, n_heads: int = 4,
                 freeze_dinov2: bool = True):
        super().__init__()
        from transformers import AutoModel
        self.dinov2 = AutoModel.from_pretrained('facebook/dinov2-base')
        self.freeze_dinov2 = freeze_dinov2
        if freeze_dinov2:
            for p in self.dinov2.parameters():
                p.requires_grad = False
            self.dinov2.eval()
        d_dino = self.dinov2.config.hidden_size   # 768

        self.proj = nn.Linear(d_dino, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                               batch_first=True, activation='gelu')
        self.ctx = nn.TransformerEncoder(enc_layer, num_layers=n_ctx_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.w_downsample = _TILE // _GRID   # px per output column, interface parity with D1

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_dinov2:
            self.dinov2.eval()
        return self

    def forward(self, strip: torch.Tensor) -> torch.Tensor:
        """strip: (1, 1, H, W), any H (upsampled internally to the tile size).
        Returns (W_col, d_model) L2-normalized, W_col = n_tiles * _GRID."""
        tiles, n_tiles = strip_to_tiles(strip, tile=_TILE)          # (n_tiles, 3, 224, 224)
        ctx = torch.no_grad() if self.freeze_dinov2 else torch.enable_grad()
        with ctx:
            out = self.dinov2(pixel_values=tiles).last_hidden_state   # DINOv2 interpolates pos-emb internally
        patches = out[:, 1:, :]                                       # (n_tiles, 256, 768), drop CLS
        patches = patches.reshape(n_tiles, _GRID, _GRID, -1)          # (n_tiles, 16, 16, 768)
        per_col = patches.mean(dim=1)                                  # (n_tiles, 16, 768) pool over height
        per_col = per_col.reshape(n_tiles * _GRID, -1)                # (W_col, 768)

        x = self.proj(per_col)                                         # (W_col, d_model)
        x = x + _sinusoidal_pe(x.shape[0], self.d_model, x.device)
        x = self.ctx(x.unsqueeze(0)).squeeze(0)
        x = self.out_proj(x)
        return F.normalize(x, dim=-1)
