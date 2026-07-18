"""DINOv2VisualNeck: replaces ConditionalUNet's entire from-scratch 4-stage
encoder (raw-pixel Conv2d downsampling + skip connections) with a frozen-
DINOv2-derived feature pyramid. This is the genuine architectural change
this project's other experiments are not -- B2-B6/C2 only add training
losses (same inference graph as A0), and B1a/MERT+X only swap the audio
encoder (same "one embedding -> FiLM everywhere" interface). This changes
the network's actual computational structure on the visual side.

Known, accepted risk (flagged before building this): the precomputed tile
grid (scripts/precompute_dinov2_tiled_native.py, 224x224 native-resolution
tiles, e.g. 6x4=24 tokens for a 1181x835 page) is far coarser than the
finest encoder stage's native resolution (e.g. ~393x278 for the first
stage) -- every stage's feature map is an interpolation of the SAME ~24
coarse tokens, not independently-resolved fine detail the way the original
from-scratch encoder's early conv layers see raw pixels directly. This is
tested empirically, not assumed to fail or succeed.

Design: one shared Linear(768, 64) projects every tile token, then EACH
required (channels, H, W) output (4 encoder residuals + 1 bottleneck input)
gets its own bilinear resize of the shared projected grid to that exact
resolution, followed by a small per-stage Conv2d+GroupNorm+ELU adapter --
mirrors ConditionalUNetBlock's own conv1/norm1/activation convention so the
downstream FiLM+decoder sees feature statistics it was designed for.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _StageAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = F.elu(self.norm(self.conv(x)))
        return x


class DINOv2VisualNeck(nn.Module):
    """stage_channels: output channel count for each of the 4 encoder-
    replacement residuals, in the SAME order/convention as
    ConditionalUNet's own encoder stages (residuals[0] = finest/largest
    resolution). bottleneck_channels: channel count for the tensor fed into
    the UNCHANGED bottleneck_block (must match its own `in_channels`,
    i.e. the original encoder's final stage `out_` -- same value as
    stage_channels[-1] in the stock architecture)."""

    def __init__(self, stage_channels: list[int], bottleneck_channels: int, d_dinov2: int = 768):
        super().__init__()
        self.proj = nn.Linear(d_dinov2, 64)
        self.stage_adapters = nn.ModuleList([_StageAdapter(64, c) for c in stage_channels])
        self.bottleneck_adapter = _StageAdapter(64, bottleneck_channels)

    def forward(self, dinov2_grid: torch.Tensor, stage_sizes: list[tuple[int, int]],
               bottleneck_size: tuple[int, int]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """dinov2_grid: (bs, n_rows, n_cols, 768), constant per piece (same
        grid broadcast across every frame of that piece -- caller's
        responsibility, matching how `score` is already constant per piece).
        stage_sizes: list of (H, W) target resolutions, one per residual,
        finest-first (matching residuals[0..n_encoder_layers-1] order).
        Returns (residuals, bottleneck_input)."""
        bs = dinov2_grid.shape[0]
        grid = dinov2_grid.permute(0, 3, 1, 2)   # (bs, 768, n_rows, n_cols)
        grid = self.proj(grid.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # (bs, 64, n_rows, n_cols)

        residuals = [adapter(grid, size) for adapter, size in zip(self.stage_adapters, stage_sizes)]
        bottleneck_input = self.bottleneck_adapter(grid, bottleneck_size)
        return residuals, bottleneck_input
