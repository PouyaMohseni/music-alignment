"""S1 audio tower: the SMALL one, deliberately.

Every larger audio representation we tried lost on room: 768-dim MERT 56.6,
+cross-attention 35.3 and 19.3, +DINOv2 2.6, against the native 78-band mel at
67.1. Part of that is confounded with the augmentation those runs lost by
precomputing -- but nothing in our data argues for MORE audio capacity on 353
training pieces, and this is trained on waveforms so the augmentation is back.

Mirrors CYOLO's proven shape: conv stack over a 40-frame x 78-band window ->
LSTM over the window sequence -> z. The `concat(hidden, last_step)` trick is
kept: it gives the conditioning a direct non-recurrent path alongside the
recurrent one, which is what lets the model react to the current frame without
waiting for the LSTM state to turn over.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class StripAudioEncoder(nn.Module):
    def __init__(self, n_mels: int = 78, zdim: int = 128, spec_out: int = 32,
                 hidden: int = 64, n_layers: int = 1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.GroupNorm(1, 24), nn.ELU(False),
            nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(1, 24), nn.ELU(False),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GroupNorm(1, 48), nn.ELU(False),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(1, 48), nn.ELU(False),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.GroupNorm(1, 96), nn.ELU(False),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(96 * 4, spec_out), nn.LayerNorm(spec_out), nn.ELU(False),
        )
        self.rnn = nn.LSTM(spec_out, hidden, num_layers=n_layers, batch_first=True)
        self.z_enc = nn.Sequential(
            nn.Linear(hidden + spec_out, zdim), nn.LayerNorm(zdim), nn.ELU(False))
        self.spec_out = spec_out

    def forward(self, mel: torch.Tensor, hidden=None):
        """mel: (N, T, n_mels) -> z: (N, zdim)."""
        n, t, f = mel.shape
        e = self.enc(mel[:, None])               # (N, spec_out) over whole window
        seq = e[:, None, :]                      # (N, 1, spec_out)
        out, hidden = self.rnn(seq, hidden)
        z = self.z_enc(torch.cat([out[:, -1], e], dim=-1))
        return z, hidden
