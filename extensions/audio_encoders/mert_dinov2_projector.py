"""MERTDINOv2Projector: low-risk visual-pretraining experiment. Same
structure as MERTProjector (extensions/audio_encoders/mert_projector.py),
just doubled input width -- consumes the concatenated (1536,) vector
extensions/hooks/mert_dinov2_patch.py produces (768 MERT audio + 768 DINOv2
CLS, broadcast constant across all frames of a piece). Zero changes to
network.py/iterate_dataset/call signatures; this is purely a wider input
Linear layer, everything else (FiLM, LSTM, decoder, skip connections) is
completely untouched -- the deliberately low-risk half of the visual-
pretraining experiment (see mert_dinov2_patch.py's docstring for why the
higher-risk full-encoder-replacement variant is a separate, bigger effort).
"""
import torch
import torch.nn as nn


class MERTDINOv2Projector(nn.Module):
    def __init__(self, spec_enc, d_mert=768, d_visual=768):
        super().__init__()
        self.n_input_frames = 1
        self.d_mert = d_mert
        self.d_visual = d_visual
        self.d_total = d_mert + d_visual
        self.enc = nn.Sequential(
            nn.Linear(self.d_total, 256),
            nn.ELU(),
            nn.Linear(256, spec_enc),
        )
        self.means = nn.Parameter(torch.zeros(self.d_total, 1), requires_grad=False)
        self.stds  = nn.Parameter(torch.ones(self.d_total, 1), requires_grad=False)

    def set_stats(self, means, stds):
        self.means = nn.Parameter(torch.from_numpy(means).view(self.d_total, 1), requires_grad=False)
        self.stds  = nn.Parameter(torch.from_numpy(stds).view(self.d_total, 1), requires_grad=False)

    def reshape_input(self, x):
        seq_len, bs, c, h, w = x.shape   # (S, B, 1, 1536, 1)
        return x.view(seq_len * bs, -1)   # (SB, 1536)

    def forward(self, x):
        x = self.reshape_input(x)
        x = (x - self.means.view(1, -1)) / self.stds.view(1, -1)
        return self.enc(x)
