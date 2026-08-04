"""Swap the frozen `(x - means)/stds` step in the audio tower for the
test-time adaptive estimate in extensions/audio_encoders/adaptive_norm.py.

Patches the FORWARD ONLY -- no parameters are added, removed or renamed, so
every existing checkpoint loads unchanged and this is a zero-retrain
intervention. `self.means`/`self.stds` stay exactly as the checkpoint stored
them and are still used: they are the `alpha=0` end of the blend and, in
`mean` mode, still supply the scale.

Covers both audio towers:
  * CBEncoder      -- 78 log-mel bands, statistics taken over time within the
                      forward call (the theoretically motivated case: a static
                      room/mic gain is an additive per-band offset in log-mel,
                      see adaptive_norm.py).
  * MERTProjector  -- 768 MERT dims. NOT a log-magnitude space, so the exact
                      channel-cancellation argument does not transfer; this is
                      the weaker "remove the domain mean shift" version and is
                      reported separately for that reason.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from extensions.audio_encoders.adaptive_norm import adapt_stats


def _cfg():
    return dict(
        mode=os.environ.get('ADAPTNORM_MODE', 'mean'),
        alpha=float(os.environ.get('ADAPTNORM_ALPHA', '1.0')),
        var_shrink=float(os.environ.get('ADAPTNORM_VAR_SHRINK', '0.5')),
    )


def patch_adaptive_input_norm(mode=None, alpha=None, var_shrink=None):
    cfg = _cfg()
    if mode is not None:
        cfg['mode'] = mode
    if alpha is not None:
        cfg['alpha'] = alpha
    if var_shrink is not None:
        cfg['var_shrink'] = var_shrink

    from audio_conditioned_unet import audio_encoder as cpjku_audio_encoder

    def cb_forward(self, x):
        x = self.reshape_input(x)                       # (SB, c, 78, 40)
        n_bands = x.shape[-2]
        mg = self.means.reshape(-1)
        sg = self.stds.reshape(-1)
        # (SB, c, F, T) -> (SB*c*T, F): one observation per (frame, window pos),
        # so the mean is taken over TIME within this chunk, per band.
        obs = x.permute(0, 1, 3, 2).reshape(-1, n_bands)
        m, s = adapt_stats(obs, mg, sg, **cfg)
        x = (x - m.view(1, 1, -1, 1)) / s.view(1, 1, -1, 1)
        return F.elu(self.norm(self.enc(x)))

    cpjku_audio_encoder.CBEncoder.forward = cb_forward
    patched = ['CBEncoder']

    MERTProjector = getattr(cpjku_audio_encoder, 'MERTProjector', None)
    if MERTProjector is not None:
        def mert_forward(self, x):
            x = self.reshape_input(x)                   # (SB, 768)
            mg = self.means.reshape(-1)
            sg = self.stds.reshape(-1)
            m, s = adapt_stats(x, mg, sg, **cfg)
            return self.enc((x - m.view(1, -1)) / s.view(1, -1))

        MERTProjector.forward = mert_forward
        patched.append('MERTProjector')

    print(f'[adaptive_norm_patch] patched {"+".join(patched)} forward: '
          f'mode={cfg["mode"]} alpha={cfg["alpha"]} var_shrink={cfg["var_shrink"]}',
          flush=True)
