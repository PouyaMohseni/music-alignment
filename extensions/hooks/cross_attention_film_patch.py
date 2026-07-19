"""Patches audio_conditioned_unet.network.ConditionalUNetBlock so every FiLM
site uses SpatialCrossAttentionFiLM (extensions/heads/cross_attention_film.py)
instead of stock FiLM. Everything else in ConditionalUNet -- the encoder/
decoder conv stack, skip connections, bottleneck, LSTM, audio_encoder choice
-- is completely untouched, so this composes with any audio-encoder patch
(e.g. extensions/hooks/mert_patch.py for B1a) applied before or after it.

NOT a subclass of the original ConditionalUNetBlock: its __init__ contains
`super(ConditionalUNetBlock, self).__init__()`, an explicit two-arg super()
that resolves `ConditionalUNetBlock` as a GLOBAL NAME in network.py's own
module namespace at call time -- once that name is monkey-patched, the
original method body's own super() call would resolve to the subclass
itself and recurse (verified: raises "missing in_channels/out_channels").
So this is a fully independent nn.Module, structurally identical to the
original (same conv1/conv2/norm1/norm2/up_conv/max_pool/forward logic,
copied verbatim), with only the FiLM line swapped -- using Python 3's
zero-arg super() (bound to this class's own __class__ cell at definition
time, immune to later global reassignment).
"""
from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


def patch_cross_attention_film():
    from audio_conditioned_unet import network as cpjku_network
    from extensions.heads.cross_attention_film import SpatialCrossAttentionFiLM

    pad = cpjku_network.pad   # unchanged helper (residual-size padding on upsample)

    class ConditionalUNetBlockCrossAttn(nn.Module):
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
                self.film_layer = SpatialCrossAttentionFiLM(spec_out, out_channels)

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

    cpjku_network.ConditionalUNetBlock = ConditionalUNetBlockCrossAttn
    print('[cross_attention_film_patch] Replaced FiLM with SpatialCrossAttentionFiLM '
          '(audio query attends over the block\'s own spatial feature map) in '
          'ConditionalUNetBlock', flush=True)
