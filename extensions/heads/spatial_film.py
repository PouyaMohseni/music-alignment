"""SpatialFiLM: SPADE-inspired spatially-varying variant of stock FiLM
(third_party/cpjku_unet/audio_conditioned_unet/network.py's FiLM class).

Stock FiLM predicts ONE (gamma, beta) pair per channel from the audio
embedding and broadcasts it identically to every pixel in the feature map --
context-blind to what's actually at each spatial location. Park et al.,
"Semantic Image Synthesis with Spatially-Adaptive Normalization" (SPADE,
2019) fixed exactly this for image-conditioned generation by predicting a
full (channels, H, W) gamma/beta MAP instead of a per-channel vector, via a
small conv net over the (spatial) conditioning signal.

Our conditioning signal (the audio embedding) has no spatial shape to
convolve, unlike SPADE's segmentation mask -- so this reshapes the audio
vector into a small COARSE grid (one learned linear projection), then
upsamples + convolves that grid to the block's actual resolution. This gets
genuine per-position variation in gamma/beta (unlike stock FiLM, which is
mathematically incapable of it) without needing a real attention/token
mechanism (unlike the cross-attention experiments) -- a cheaper, weaker
middle ground: spatial variation with no actual content-awareness of the
image, only a learned coarse pattern.
"""
from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


class SpatialFiLM(nn.Module):
    def __init__(self, zdim: int, maskdim: int, coarse_size: tuple[int, int] = (4, 4)):
        super().__init__()
        self.maskdim = maskdim
        self.coarse_h, self.coarse_w = coarse_size
        self.to_grid = nn.Linear(zdim, maskdim * self.coarse_h * self.coarse_w)
        self.refine_gamma = nn.Conv2d(maskdim, maskdim, kernel_size=3, padding=1)
        self.refine_beta = nn.Conv2d(maskdim, maskdim, kernel_size=3, padding=1)

    def forward(self, x, z):
        b, c, h, w = x.shape
        grid = self.to_grid(z).view(b, self.maskdim, self.coarse_h, self.coarse_w)
        grid_up = F.interpolate(grid, size=(h, w), mode='bilinear', align_corners=False)
        gamma = self.refine_gamma(grid_up)
        beta = self.refine_beta(grid_up)
        return gamma * x + beta
