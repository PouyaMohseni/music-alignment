"""R2a -- channel augmentation in MERT EMBEDDING space (no re-encoding).

The nuisance R1 identifies is a static channel: constant over a recording,
varying between recordings. In log-mel that is a per-band offset. MERT's 768-d
space is not log-magnitude, so the effect is not exactly an offset there, but
the defining property survives -- it is a low-frequency, time-CONSTANT
perturbation of the feature vector, not per-frame noise.

So this draws a per-dimension affine ONCE PER FORWARD CALL and holds it across
the whole chunk:

    x' = x * exp(s) + b*sigma        s ~ N(0, sigma_s^2)   per dim
                                     b ~ N(0, sigma_b^2)   per dim

`sigma` is the checkpoint's own per-dimension training std, so sigma_b is
dimensionless and means "shift this dimension by N training-standard-
deviations" rather than depending on MERT's arbitrary scale.

TIME-CONSTANT IS THE WHOLE POINT. Resampling per frame would be ordinary
input noise, which the model can average away over the 40-frame window and
which teaches nothing about channels. Held constant, the only way to be
invariant to it is to stop relying on absolute per-dimension levels -- which
is exactly the invariance a room mic demands.

WHY THIS EXISTS ALONGSIDE R2 (waveform augmentation + MERT re-encode). This is
a proxy: it perturbs MERT's OUTPUT, whereas a real room perturbs its INPUT,
and a frozen nonlinear encoder does not commute with the two. R2 is the
faithful version and costs 6615 renders. Running both answers whether that
cost buys anything, which is a result either way -- and unlike R2 this one can
start immediately.

TRAIN ONLY: gated on self.training, so eval is bit-for-bit unaugmented.
"""
from __future__ import annotations

import os

import torch


def patch_mert_channel_aug(sigma_s=None, sigma_b=None, p=None):
    sigma_s = float(os.environ.get('CHANAUG_SIGMA_S', '0.15')) if sigma_s is None else sigma_s
    sigma_b = float(os.environ.get('CHANAUG_SIGMA_B', '0.35')) if sigma_b is None else sigma_b
    p = float(os.environ.get('CHANAUG_P', '0.8')) if p is None else p

    from audio_conditioned_unet import audio_encoder as cpjku_audio_encoder

    MERTProjector = getattr(cpjku_audio_encoder, 'MERTProjector', None)
    if MERTProjector is None:
        raise RuntimeError('MERTProjector is not registered on the audio_encoder module -- '
                           'apply the MERT pipeline patch before this one')

    def forward(self, x):
        x = self.reshape_input(x)                        # (SB, 768)
        mg = self.means.view(1, -1)
        sg = self.stds.view(1, -1)

        if self.training and torch.rand(()) < p:
            # one draw per call, broadcast over all SB frames -> time-constant
            s = torch.randn(1, x.shape[-1], device=x.device, dtype=x.dtype) * sigma_s
            b = torch.randn(1, x.shape[-1], device=x.device, dtype=x.dtype) * sigma_b
            x = x * torch.exp(s) + b * sg

        return self.enc((x - mg) / sg)

    MERTProjector.forward = forward
    print(f'[channel_aug_patch] MERTProjector channel augmentation: '
          f'sigma_s={sigma_s} sigma_b={sigma_b} p={p} (train only, one draw per chunk)',
          flush=True)
