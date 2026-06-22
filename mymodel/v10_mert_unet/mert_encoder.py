"""MERTProjector — drop-in replacement for CBEncoder in ConditionalUNet.

Projects precomputed MERT-v1-95M frame embeddings (768-dim) to spec_enc-dim.
Input shape mirrors CBEncoder: (seq_len, B, 1, 768, 1).
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
        self.means = nn.Parameter(torch.zeros(d_mert, 1), requires_grad=False)
        self.stds  = nn.Parameter(torch.ones(d_mert, 1), requires_grad=False)

    def set_stats(self, means, stds):
        pass  # MERT embeddings are already normalized internally

    def reshape_input(self, x):
        seq_len, bs, c, h, w = x.shape   # (S, B, 1, 768, 1)
        return x.view(seq_len * bs, -1)   # (SB, 768)

    def forward(self, x):
        x = self.reshape_input(x)   # (SB, 768)
        return self.enc(x)          # (SB, spec_enc)
