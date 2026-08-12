"""H1 -- MERT audio tower inside CYOLO's detector.

WHY THIS COMBINATION. The two largest effects we have measured live on opposite
sides of the architecture and have never been combined:

  * The DETECTION output parameterisation is worth a great deal on real audio.
    Our own reproduction, same IR bank and protocol as our heatmap model,
    reaches 67.1 on `room` at 18% of training where the heatmap model converges
    at 56.6.
  * The MERT audio tower is worth ~+22 on `room` over the 78-band mel CNN,
    measured across our own sweep.

CYOLO's robustness comes from its OUTPUT side -- boxes, objectness, the
multi-scale head -- none of which cares what produced the conditioning vector
`z`. Its INPUT side is a plain mel-spectrogram CNN, which is precisely the
component our sweep shows is worth -22 on room. The two contributions are
architecturally independent, so they should compose.

WHAT IS AND IS NOT REPLACED. `ContextConditioning` windows the performance into
kw=40-frame chunks, encodes each chunk to `spec_out` dims via `self.enc`, runs
an LSTM over the chunk sequence, and forms
`z = z_enc(concat(lstm_hidden, last_steps))`. ONLY `self.enc` is replaced. The
kw=40 windowing, the LSTM, the concat trick, the FPN, the anchors and the `sb`
multi-class head are all untouched, so any change in the result is attributable
to the audio representation.

SHAPE CONTRACT. The original enc maps (N, 1, 40, 78) -> (N, spec_out). This one
maps (N, 1, 40, 768) -> (N, spec_out), so it is a drop-in and every downstream
shape is unchanged.

DESIGN. A per-frame linear projection 768 -> `proj_dim` (MERT frames are
already rich; the expensive part should not be a 2-D conv over a 768-tall
"image"), then a small temporal conv stack over the 40 frames, mirroring the
original's conv -> flatten -> linear -> norm -> activation structure so the
window's temporal texture is preserved rather than being average-pooled away.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MERTWindowEncoder(nn.Module):
    """(N, 1, KW, 768) -> (N, spec_out). Drop-in for ContextConditioning.enc."""

    def __init__(self, in_dim: int = 768, spec_out: int = 32, proj_dim: int = 128,
                 kw: int = 40, activation=nn.ELU, groupnorm: bool = False):
        super().__init__()
        act = activation
        self.in_dim = in_dim
        self.kw = kw

        # Per-frame projection. LayerNorm first: MERT features are not
        # zero-mean/unit-variance and the downstream BatchNorms behave much
        # better if the projection sees a normalised input.
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
            act(False),
        )

        def norm1d(c):
            return nn.GroupNorm(1, c) if groupnorm else nn.BatchNorm1d(c)

        # Temporal stack over the 40 frames: 40 -> 20 -> 10 -> 5
        self.temporal = nn.Sequential(
            nn.Conv1d(proj_dim, 96, kernel_size=3, padding=1), norm1d(96), act(False),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 96, kernel_size=3, padding=1), norm1d(96), act(False),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 96, kernel_size=3, padding=1), norm1d(96), act(False),
            nn.MaxPool1d(2),
        )
        n_flat = 96 * (kw // 8)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, spec_out),
            nn.LayerNorm(spec_out) if groupnorm else nn.BatchNorm1d(spec_out),
            act(False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, KW, in_dim)
        if x.dim() != 4 or x.shape[-1] != self.in_dim:
            raise ValueError(f'expected (N,1,KW,{self.in_dim}), got {tuple(x.shape)}')
        x = x[:, 0]                       # (N, KW, in_dim)
        x = self.proj(x)                  # (N, KW, proj_dim)
        x = x.transpose(1, 2)             # (N, proj_dim, KW)
        x = self.temporal(x)              # (N, 96, KW/8)
        return self.head(x)               # (N, spec_out)


def resample_frames(emb: torch.Tensor, factor: float) -> torch.Tensor:
    """Tempo augmentation for a PRECOMPUTED embedding sequence.

    CYOLO augments tempo with a phase vocoder on the waveform, which we cannot
    do once MERT has been computed offline.  Stretching the frame axis is the
    available approximation: it reproduces the effect the augmentation exists
    for -- forcing the LSTM to be invariant to how fast the score is traversed
    -- without re-rendering the acoustics.  It is NOT equivalent to a phase
    vocoder, and that difference is worth stating in any writeup.

    emb: (T, D) -> (T', D) with T' = round(T / factor).
    """
    if abs(factor - 1.0) < 1e-3 or emb.shape[0] < 2:
        return emb
    t_new = max(2, int(round(emb.shape[0] / factor)))
    x = emb.transpose(0, 1).unsqueeze(0)                       # (1, D, T)
    x = torch.nn.functional.interpolate(x, size=t_new, mode='linear', align_corners=False)
    return x.squeeze(0).transpose(0, 1).contiguous()
