"""GatedSpatialFiLM: combines SpatialFiLM's mechanism (extensions/heads/
spatial_film.py -- SPADE-inspired spatially-varying gamma/beta from a coarse
learned grid) with GatedFiLM's AdaLN-Zero-style zero-initialized gate
(extensions/heads/gated_film.py).

Motivation: completes the mechanism x gating ablation grid started by
gated_cross_attention_film.py. B1a-spatial-film was the worst of the three
FiLM replacements eval'd 2026-07-27/28 (44.3% pct@0.5s, vs cross-attention's
71.1% and gated-film's 82.9%, all starting from the same B1a/MERT audio
encoder) -- and, like cross-attention FiLM, it was applied at full
random-initialized strength from step one. This module tests whether
spatial FiLM's much larger underperformance was similarly a stabilization
problem rather than a genuine mechanism-quality problem: same coarse-grid
spatial modulation, but blended in via a zero-initialized gate so training
starts as pure identity.

    gate = 0   -> output = x                          (pure identity)
    gate = 1   -> output = spatial_modulated(x, z)     (full strength)
"""
from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


class GatedSpatialFiLM(nn.Module):
    def __init__(self, zdim: int, maskdim: int, coarse_size: tuple[int, int] = (4, 4)):
        super().__init__()
        self.maskdim = maskdim
        self.coarse_h, self.coarse_w = coarse_size
        self.to_grid = nn.Linear(zdim, maskdim * self.coarse_h * self.coarse_w)
        self.refine_gamma = nn.Conv2d(maskdim, maskdim, kernel_size=3, padding=1)
        self.refine_beta = nn.Conv2d(maskdim, maskdim, kernel_size=3, padding=1)
        self.gate = nn.Linear(zdim, maskdim)
        # Zero-init (weight AND bias) so gate(z) == 0 for every z at
        # initialization -- same convention as GatedFiLM /
        # GatedSpatialCrossAttentionFiLM. Tagged so gated_spatial_film_patch.py's
        # patched initialize_weights re-zeros instead of orthogonal-initializing
        # (ConditionalUNet.__init__ ends with self.apply(initialize_weights)).
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.gate._gated_film_zero_init = True

    def forward(self, x, z):
        b, c, h, w = x.shape
        grid = self.to_grid(z).view(b, self.maskdim, self.coarse_h, self.coarse_w)
        grid_up = F.interpolate(grid, size=(h, w), mode='bilinear', align_corners=False)
        gamma = self.refine_gamma(grid_up)
        beta = self.refine_beta(grid_up)
        gate = self.gate(z).unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * x + beta
        return x + gate * (modulated - x)
