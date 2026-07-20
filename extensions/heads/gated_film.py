"""GatedFiLM: AdaLN-Zero-style gated variant of stock FiLM
(third_party/cpjku_unet/audio_conditioned_unet/network.py's FiLM class).

Stock FiLM applies `gamma * x + beta` at FULL STRENGTH from the very first
training step, using randomly-initialized gamma/beta -- the audio signal
immediately perturbs the image features before the network has learned
anything useful to do with it, a known source of training instability (this
exact problem is why Peebles & Xie's DiT / "Scalable Diffusion Models with
Transformers" (2022/2023) introduced AdaLN-Zero: predict a third value, a
GATE, from the same conditioning vector, zero-initialize its projection so
the gate starts at EXACTLY 0, and use it to blend the modulated output back
towards the identity function at initialization:

    gate = 0   -> output = x                      (pure identity, no audio effect)
    gate = 1   -> output = gamma * x + beta        (full stock-FiLM strength)

The network then learns, via ordinary backprop, how much to trust the audio
conditioning at each block, instead of having that decision forced on it
from step zero by a random initialization.
"""
from __future__ import annotations
import torch.nn as nn


class GatedFiLM(nn.Module):
    def __init__(self, zdim: int, maskdim: int):
        super().__init__()
        self.gamma = nn.Linear(zdim, maskdim)
        self.beta = nn.Linear(zdim, maskdim)
        self.gate = nn.Linear(zdim, maskdim)
        # Zero-init (both weight AND bias) so gate(z) == 0 for every z at
        # initialization -- no sigmoid/tanh squashing, a plain zeroed affine
        # map, matching AdaLN-Zero's own "exactly zero" convention (not 0.5,
        # which a zero-init + sigmoid would give).
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        # ConditionalUNet.__init__ ends with self.apply(initialize_weights),
        # which orthogonal-inits EVERY nn.Linear it finds -- including this
        # one, clobbering the zero-init above after construction (confirmed
        # via smoke test: gate leaked nonzero at "init" without this tag).
        # gated_film_patch.py's patched initialize_weights checks this flag
        # and re-zeros instead of orthogonal-initializing.
        self.gate._gated_film_zero_init = True

    def forward(self, x, z):
        gamma = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        gate = self.gate(z).unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * x + beta
        return x + gate * (modulated - x)
