"""Cross-attention alternative to FiLM (third_party/cpjku_unet/audio_conditioned_unet/
network.py's FiLM class). Stock FiLM: gamma,beta = Linear(audio_embedding) --
broadcast IDENTICALLY to every spatial location, blind to what is actually in
the image at that location. Score-following IS an alignment problem (find
which score position the current audio corresponds to), and attention is the
classical mechanism for exactly that (Bahdanau/Tacotron-style alignment) --
so replacing FiLM's context-blind broadcast with cross-attention (audio as
query, visual content as key/value) lets the modulation itself depend on what
is in the image, not just what is in the audio. This is a change to the
FUSION mechanism, not to which encoder produces the features (that's the
B1a/V-DINOv2 distinction).

Two variants sharing the same query/key/value/gamma/beta shape convention:
  - SpatialCrossAttentionFiLM: K/V = the block's OWN current feature map
    (flattened H*W spatial positions) -- for architectures whose visual side
    is a real per-block conv feature map (e.g. B1a's unmodified CB_TA
    encoder).
  - TokenCrossAttentionFiLM: K/V = an externally supplied, shared-across-all-
    blocks token set (e.g. the raw DINOv2 patch grid, which is already
    genuine per-patch tokens, not a per-stage conv feature map).

Both still produce ONE (gamma, beta) pair per (frame, channel), broadcast
over space exactly like stock FiLM -- the difference is WHAT gamma/beta are
a function of, not whether they vary spatially.
"""
from __future__ import annotations
import torch.nn as nn


class SpatialCrossAttentionFiLM(nn.Module):
    """zdim: audio embedding dim (query source). maskdim: this block's own
    channel count (both the K/V source -- the block's own feature map -- and
    the output gamma/beta dim)."""

    def __init__(self, zdim: int, maskdim: int, n_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(zdim, maskdim)
        self.kv_proj = nn.Conv2d(maskdim, maskdim * 2, kernel_size=1)
        self.mha = nn.MultiheadAttention(maskdim, n_heads, batch_first=True)
        self.gamma = nn.Linear(maskdim, maskdim)
        self.beta = nn.Linear(maskdim, maskdim)

    def forward(self, x, z):
        b, c, h, w = x.shape
        kv = self.kv_proj(x).flatten(2).transpose(1, 2)   # (b, hw, 2c)
        k, v = kv.chunk(2, dim=-1)
        q = self.q_proj(z).unsqueeze(1)                     # (b, 1, c)
        ctx, _ = self.mha(q, k, v)                            # (b, 1, c)
        ctx = ctx.squeeze(1)
        gamma = self.gamma(ctx).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(ctx).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class TokenCrossAttentionFiLM(nn.Module):
    """zdim: audio embedding dim (query source). maskdim: this block's own
    channel count (output gamma/beta dim). token_dim: the shared visual
    token set's own feature dim (e.g. 768 for raw DINOv2 patch tokens) --
    independent of maskdim since the SAME token set is fed to every block,
    whose maskdim varies stage to stage."""

    def __init__(self, zdim: int, maskdim: int, token_dim: int, n_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(zdim, maskdim)
        self.k_proj = nn.Linear(token_dim, maskdim)
        self.v_proj = nn.Linear(token_dim, maskdim)
        self.mha = nn.MultiheadAttention(maskdim, n_heads, batch_first=True)
        self.gamma = nn.Linear(maskdim, maskdim)
        self.beta = nn.Linear(maskdim, maskdim)

    def forward(self, x, z, visual_tokens):
        q = self.q_proj(z).unsqueeze(1)               # (b, 1, maskdim)
        k = self.k_proj(visual_tokens)                  # (b, n_tokens, maskdim)
        v = self.v_proj(visual_tokens)                  # (b, n_tokens, maskdim)
        ctx, _ = self.mha(q, k, v)                        # (b, 1, maskdim)
        ctx = ctx.squeeze(1)
        gamma = self.gamma(ctx).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(ctx).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta
