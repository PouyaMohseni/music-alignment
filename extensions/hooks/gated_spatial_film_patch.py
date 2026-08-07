"""Patches audio_conditioned_unet.network.ConditionalUNetBlock so every FiLM
site uses GatedSpatialFiLM (extensions/heads/gated_spatial_film.py) instead
of stock FiLM. Everything else in ConditionalUNet is untouched, so this
composes with any audio-encoder patch (e.g. extensions/hooks/mert_patch.py
for B1a) applied before or after it.

Completes the mechanism x gating ablation grid: same coarse-grid spatial
modulation as spatial_film_patch.py, but blended in via a zero-initialized
gate exactly like gated_film_patch.py / gated_cross_attention_film_patch.py.

NOT a subclass of the original ConditionalUNetBlock -- see
cross_attention_film_patch.py's docstring for why: the original __init__
calls `super(ConditionalUNetBlock, self).__init__()`, an explicit two-arg
super() that resolves `ConditionalUNetBlock` as a GLOBAL NAME in network.py's
own module namespace at call time, so once that name is monkey-patched, the
original method's own super() call recurses into the subclass instead of
reaching nn.Module. This is a fully independent nn.Module, structurally
identical to the original, with only the FiLM line swapped.
"""
from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


def patch_gated_spatial_film():
    from audio_conditioned_unet import network as cpjku_network
    from extensions.heads.gated_spatial_film import GatedSpatialFiLM

    pad = cpjku_network.pad

    # Same zero-init-preservation wrap as gated_film_patch.py /
    # gated_cross_attention_film_patch.py.
    _orig_initialize_weights = cpjku_network.initialize_weights

    def _initialize_weights_preserve_gate(m):
        if getattr(m, '_gated_film_zero_init', False):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            return
        _orig_initialize_weights(m)

    cpjku_network.initialize_weights = _initialize_weights_preserve_gate

    class ConditionalUNetBlockGatedSpatialFiLM(nn.Module):
        def __init__(self, in_channels, out_channels, spec_out=128, film=True, down_sample=True,
                     up_sample=False, up_in_channels=1, padding=1, no_skip=False):
            super().__init__()

            self.up_sample = up_sample
            self.down_sample = down_sample
            self.film = film
            self.no_skip = no_skip
            self.in_channels = in_channels

            if self.up_sample:
                self.up_conv = nn.Sequential(nn.Upsample(scale_factor=2),
                                             nn.Conv2d(up_in_channels, in_channels, kernel_size=1, stride=1))
            if self.down_sample:
                self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding)
            self.norm1 = nn.GroupNorm(1, out_channels)
            self.norm2 = nn.GroupNorm(1, out_channels)

            if self.film:
                self.film_layer = GatedSpatialFiLM(spec_out, out_channels)

        def forward(self, x, spec, residual=None):
            if self.up_sample:
                x = self.up_conv(x)
                if residual is not None:
                    x = pad(x, residual.size())
                    if not self.no_skip:
                        x = x + residual

            x = F.elu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))

            if self.film:
                x = self.film_layer(x, spec)

            x = F.elu(x)

            if self.down_sample:
                return x, self.max_pool(x)
            else:
                return x

    cpjku_network.ConditionalUNetBlock = ConditionalUNetBlockGatedSpatialFiLM
    print('[gated_spatial_film_patch] Replaced FiLM with GatedSpatialFiLM (zero-init gated '
          'spatially-varying gamma/beta) in ConditionalUNetBlock', flush=True)
