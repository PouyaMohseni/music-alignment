"""A5 -- adaLN-Zero conditioning, a drop-in replacement for FiLM.

WHY THIS ONE AND NOT ANOTHER
----------------------------
We asked whether any conditioning mechanism consistently beats FiLM. The
literature says no:

  * A seven-way controlled ablation on one backbone -- Cross-Attention, Prefix
    tuning, FiLM, AdaLN, AdaLN-Zero, adaRMSNorm, Additive injection -- found all
    seven COMPARABLE under normal training (Decoupled Action Expert, 2024).
  * adaLN-Zero consistently beats cross-attention and in-context conditioning
    (DiT ablations, FID and compute), but versus FiLM it is context-dependent.
  * FiLM beat four alternatives including cross-attention variants in SAINT
    (2025); AdaLN clearly beat prepending in Dyadic Mamba (2025).

So this is expected to be worth a couple of points at most, and is run as an
ablation rather than as a bet. It is the ONE variant worth the cheap test, for
three reasons that all follow from our own numbers:

  1. Cross-attention is the mechanism the literature says loses at small scale,
     and we measured exactly that: 19.3 on room for B1a_cross_attention and 2.6
     for MERT_dinov2_cross_attention, against 38.5 for plain FiLM.
  2. adaLN-Zero is FiLM-SHAPED -- channel-wise scale and shift -- and FiLM is
     the only conditioning that has ever worked here.
  3. The ZERO INITIALISATION is the part our failed cross-attention runs lacked.
     Those had to learn a whole new pathway from random init on a few hundred
     pieces. This starts as an exact no-op and learns modulation gradually, so
     it cannot collapse the way they did.

WHAT IT CHANGES relative to audio_conditioned_unet.network.FiLM, which does

    x = gamma(z) * x + beta(z)

with gamma/beta randomly initialised. Two differences:

  * The feature is NORMALISED before modulation (affine-free GroupNorm), so the
    scale the conditioner has to produce is standardised across layers and
    depths. This is what makes adaptive normalisation better-conditioned than
    raw FiLM.
  * The projection producing (gamma, beta) is ZERO-INITIALISED, and the scale
    enters as (1 + gamma). At step 0 the layer therefore returns exactly
    norm(x), i.e. the unconditioned feature, and every gradient the conditioner
    receives is earned rather than fighting a random initial modulation.

NOTE ON WHAT "IDENTITY" MEANS HERE. DiT's adaLN-Zero also zero-gates a residual
branch, making the whole block an exact identity at init. FiLM in this network
is not residual -- it REPLACES the feature -- so the strongest available
statement is "passes the normalised feature through unmodulated at init". That
difference is real and is why this is not literally DiT's block.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """Drop-in for FiLM(zdim, maskdim): forward(x, z) -> modulated x.

    x: (N, C, H, W) with C == maskdim;  z: (N, zdim).
    """

    def __init__(self, zdim: int, maskdim: int, norm: str = 'group'):
        super().__init__()
        self.maskdim = maskdim
        if norm == 'group':
            # affine=False: the affine part is exactly what z provides
            self.norm = nn.GroupNorm(1, maskdim, affine=False)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(maskdim, affine=False)
        else:
            raise ValueError(f'unknown norm {norm!r}')
        self.proj = nn.Linear(zdim, 2 * maskdim)
        # THE point of the method: start as a no-op.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(z).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1.0 + gamma) + beta


def patch_adaln_zero(norm: str = 'group'):
    """Replace audio_conditioned_unet.network.FiLM with AdaLNZero.

    Patches the CLASS so every FiLM site in the UNet picks it up at
    construction. Signature and call convention are identical, so nothing else
    in the network changes -- which is what makes this an isolated ablation of
    the conditioning mechanism.
    """
    from audio_conditioned_unet import network as net_mod

    if getattr(net_mod, '_adaln_zero_patched', False):
        return

    _orig = net_mod.FiLM

    class _FiLMShim(AdaLNZero):
        def __init__(self, zdim, maskdim):
            super().__init__(zdim, maskdim, norm=norm)

    net_mod.FiLM = _FiLMShim
    net_mod._adaln_zero_patched = True
    net_mod._orig_FiLM = _orig
    print(f'[A5] FiLM -> AdaLNZero (norm={norm}, zero-init projection) ACTIVE',
          flush=True)
