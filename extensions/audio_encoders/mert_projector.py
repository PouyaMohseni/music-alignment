"""B1 -- MERTProjector: drop-in replacement for CBEncoder in ConditionalUNet.

Matches CBEncoder's interface exactly (n_input_frames, forward(x)->(SB,spec_enc),
set_stats) so it plugs into CPJKU's unmodified network.py via
`getattr(audio_encoder_module, config['audio_encoder'])(spec_enc)` -- see
extensions/hooks/mert_patch.py for how this gets registered.

Input shape mirrors CBEncoder: (seq_len, B, 1, 768, 1) -- one precomputed
MERT embedding per 20fps timestep, no windowing needed (unlike CBEncoder's
40-frame raw-spectrogram window) since MERT already aggregates temporal
context internally from its own pretraining.
"""
import torch
import torch.nn as nn


class MERTProjector(nn.Module):
    def __init__(self, spec_enc, d_mert=768):
        super().__init__()
        self.n_input_frames = 1
        self.d_mert = d_mert
        self.enc = nn.Sequential(
            nn.Linear(d_mert, 256),
            nn.ELU(),
            nn.Linear(256, spec_enc),
        )
        # kept for CBEncoder-interface parity; MERT embeddings are normalized
        # via set_stats using train-set mean/std, same mechanism as CBEncoder
        self.means = nn.Parameter(torch.zeros(d_mert, 1), requires_grad=False)
        self.stds  = nn.Parameter(torch.ones(d_mert, 1), requires_grad=False)

    def set_stats(self, means, stds):
        self.means = nn.Parameter(torch.from_numpy(means).view(self.d_mert, 1), requires_grad=False)
        self.stds  = nn.Parameter(torch.from_numpy(stds).view(self.d_mert, 1), requires_grad=False)

    def reshape_input(self, x):
        seq_len, bs, c, h, w = x.shape   # (S, B, 1, 768, 1)
        return x.view(seq_len * bs, -1)   # (SB, 768)

    def forward(self, x):
        x = self.reshape_input(x)              # (SB, 768)
        x = (x - self.means.view(1, -1)) / self.stds.view(1, -1)
        return self.enc(x)                      # (SB, spec_enc)
