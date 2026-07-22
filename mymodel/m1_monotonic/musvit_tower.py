"""MuSViTScoreTower -- frozen MuSViT (music-notation ViT, PRAIG/musvit, MAE-
pretrained on 9.7M IMSLP pages) as an alternative to D1's from-scratch
ScoreTower, matching the SAME output interface: strip -> (W_col, d_model)
L2-normalized per-column embeddings.

MuSViT was pretrained on square-ish PAGES (1024x1024 / 512x512), never on our
horizontal single-line STRIP format -- a raw whole-strip resize badly distorts
its 13:1 aspect ratio (confirmed: squashed-square features nearly collapse,
cosine ~0.965). Fix: tile the strip NATIVELY at its own height (no vertical
squash) -- non-overlapping 224x224 crops slid along width -- run frozen MuSViT
per tile, take the per-tile 14x14 patch grid, pool over height -> per-column
raw features, then a SMALL TRAINABLE projection + transformer context layer
(identical role to D1's own ScoreTower context stage) turns raw frozen
appearance features into a position-discriminative representation. This
mirrors D1's own design (positional encoding + transformer are what create
position-discrimination, not the raw backbone features) and is the correct bar
to test MuSViT against -- not raw frozen cosine similarity (see M1.md).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_TILE = 224            # MuSViT patch=16, no vertical squash (matches native strip height)
_PATCH = 16
_GRID = _TILE // _PATCH   # 14
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _sinusoidal_pe(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe


def strip_to_tiles(strip_01: torch.Tensor, tile: int = _TILE) -> tuple[torch.Tensor, int]:
    """strip_01: (1, 1, H, W) in [0,1] (D1's own strip convention: 1 - sheet/255).
    D1's own strip is downscaled by scale_factor=3 for its from-scratch CNN
    tower (e.g. 74px tall) -- far too short for MuSViT's patch=16 to resolve
    real structure. Upsample (aspect-preserving) to `tile` height first (blur
    cost accepted; keeps M1's existing data pipeline / column-space convention
    unchanged rather than threading a second, higher-res strip through). If
    already >= tile, resize down instead (keeps behavior sane either way).
    Returns (tiles, n_tiles): tiles (n_tiles, 3, tile, tile) ImageNet-normalized
    RGB (channel-replicated), n_tiles = ceil(W/tile), last tile edge-adjusted
    (overlaps its neighbor rather than padding, so every tile is full-size)."""
    _, _, H, W = strip_01.shape
    if H != tile:
        scale = tile / H
        strip_01 = F.interpolate(strip_01, size=(tile, max(1, int(round(W * scale)))),
                                 mode='bilinear', align_corners=False)
        _, _, H, W = strip_01.shape
    starts = list(range(0, max(W - tile, 0) + 1, tile))
    if not starts:
        starts = [0]
    if starts[-1] + tile < W:
        starts.append(W - tile)
    device = strip_01.device
    mean = _MEAN.to(device); std = _STD.to(device)
    tiles = []
    for s in starts:
        crop = strip_01[:, :, :, s:s + tile]                 # (1,1,tile,tile)
        rgb = crop.expand(-1, 3, -1, -1)                       # channel-replicate grayscale -> RGB
        tiles.append((rgb - mean) / std)
    return torch.cat(tiles, dim=0), len(starts)


class MuSViTScoreTower(nn.Module):
    """Frozen MuSViT backbone + trainable projection/context head. Interface-
    compatible with mymodel.d1_align_matrix.model.ScoreTower: forward(strip) ->
    (W_col, d_model) L2-normalized per-column embeddings."""

    def __init__(self, d_model: int = 128, n_ctx_layers: int = 2, n_heads: int = 4,
                 freeze_musvit: bool = True):
        super().__init__()
        from transformers import ViTModel
        self.musvit = ViTModel.from_pretrained('PRAIG/musvit', trust_remote_code=True)
        self.freeze_musvit = freeze_musvit
        if freeze_musvit:
            for p in self.musvit.parameters():
                p.requires_grad = False
            self.musvit.eval()
        d_musvit = self.musvit.config.hidden_size   # 768

        self.proj = nn.Linear(d_musvit, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                               batch_first=True, activation='gelu')
        self.ctx = nn.TransformerEncoder(enc_layer, num_layers=n_ctx_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.w_downsample = _TILE // _GRID   # px per output column, for interface parity with D1

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_musvit:
            self.musvit.eval()   # keep frozen backbone's BN/dropout in eval regardless of outer mode
        return self

    def forward(self, strip: torch.Tensor) -> torch.Tensor:
        """strip: (1, 1, H, W), H == 224 (native strip height). Returns
        (W_col, d_model) L2-normalized, W_col = n_tiles * _GRID."""
        tiles, n_tiles = strip_to_tiles(strip)                  # (n_tiles, 3, 224, 224)
        ctx = torch.no_grad() if self.freeze_musvit else torch.enable_grad()
        with ctx:
            out = self.musvit(pixel_values=tiles, interpolate_pos_encoding=True).last_hidden_state
        patches = out[:, 1:, :]                                   # (n_tiles, 196, 768), drop CLS
        patches = patches.reshape(n_tiles, _GRID, _GRID, -1)      # (n_tiles, 14, 14, 768)
        per_col = patches.mean(dim=1)                              # (n_tiles, 14, 768) pool over height
        per_col = per_col.reshape(n_tiles * _GRID, -1)            # (W_col, 768)

        x = self.proj(per_col)                                     # (W_col, d_model)
        x = x + _sinusoidal_pe(x.shape[0], self.d_model, x.device)
        x = self.ctx(x.unsqueeze(0)).squeeze(0)
        x = self.out_proj(x)
        return F.normalize(x, dim=-1)
