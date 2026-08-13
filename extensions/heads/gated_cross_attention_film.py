"""GatedSpatialCrossAttentionFiLM: combines SpatialCrossAttentionFiLM's
mechanism (extensions/heads/cross_attention_film.py -- audio query attends
over the block's own spatial feature map) with GatedFiLM's AdaLN-Zero-style
zero-initialized gate (extensions/heads/gated_film.py).

Motivation: B1a-cross-attention (71.1% pct@0.5s) underperformed B1a-gated-
film (82.9%) by a wide margin, both starting from the same B1a/MERT audio
encoder. That comparison confounds two independent things: (1) the fusion
MECHANISM (attention vs a global affine) and (2) whether the mechanism is
applied at full random-initialized strength from step one or ramped in from
identity. Gated FiLM's zero-init gate is exactly the stabilization trick
cross-attention FiLM never got. This module isolates the confound: same
cross-attention mechanism as SpatialCrossAttentionFiLM, but blended in via
a zero-initialized gate exactly like GatedFiLM, so training starts as pure
identity (no audio effect at all) and the network learns how much to trust
attention-based conditioning at each block, instead of having full-strength
untrained attention output forced on it from step zero.

    gate = 0   -> output = x                                  (pure identity)
    gate = 1   -> output = cross_attention_modulated(x, z)     (full strength)
"""
from __future__ import annotations
import torch.nn as nn


class GatedSpatialCrossAttentionFiLM(nn.Module):
    """zdim: audio embedding dim (query source). maskdim: this block's own
    channel count (both the K/V source -- the block's own feature map --
    and the output gamma/beta/gate dim)."""

    def __init__(self, zdim: int, maskdim: int, n_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(zdim, maskdim)
        self.kv_proj = nn.Conv2d(maskdim, maskdim * 2, kernel_size=1)
        self.mha = nn.MultiheadAttention(maskdim, n_heads, batch_first=True)
        self.gamma = nn.Linear(maskdim, maskdim)
        self.beta = nn.Linear(maskdim, maskdim)
        self.gate = nn.Linear(zdim, maskdim)
        # Zero-init (weight AND bias) so gate(z) == 0 for every z at
        # initialization -- same convention as GatedFiLM. See gated_film.py's
        # docstring for why plain zero (not a zero-init + sigmoid, which
        # would give 0.5) and why the _gated_film_zero_init tag is needed to
        # survive ConditionalUNet.__init__'s later self.apply(initialize_weights).
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.gate._gated_film_zero_init = True

    def forward(self, x, z):
        b, c, h, w = x.shape
        kv = self.kv_proj(x).flatten(2).transpose(1, 2)   # (b, hw, 2c)
        k, v = kv.chunk(2, dim=-1)
        q = self.q_proj(z).unsqueeze(1)                     # (b, 1, c)
        ctx, _ = self.mha(q, k, v)                            # (b, 1, c)
        ctx = ctx.squeeze(1)
        gamma = self.gamma(ctx).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(ctx).unsqueeze(-1).unsqueeze(-1)
        gate = self.gate(z).unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * x + beta
        return x + gate * (modulated - x)
