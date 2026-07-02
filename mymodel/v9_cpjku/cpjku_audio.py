"""Audio encoders from CPJKU/audio_conditioned_unet (ismir-2020 branch).

Copied verbatim — only change is removing unused imports.
CBEncoder is the context-based encoder used in their best model (CB_TA).
It takes a 40-frame (2-second) window of the 78-bin log-mel spectrogram
and encodes it to a 32-dim vector via a 2D CNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        # using only one group is equivalent to using layer norm
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        return F.elu(self.norm(self.conv(x)))


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class FBEncoder(nn.Module):
    """Frame-based encoder: single frame of 78-bin spectrogram → spec_enc dim."""
    def __init__(self, spec_enc):
        super(FBEncoder, self).__init__()
        self.n_input_frames = 1
        self.enc = nn.Linear(78, spec_enc)
        self.norm = nn.LayerNorm(spec_enc)
        self.means = nn.Parameter(torch.zeros(78), requires_grad=False)
        self.stds = nn.Parameter(torch.ones(78), requires_grad=False)

    def set_stats(self, means, stds):
        self.means = nn.Parameter(torch.from_numpy(means), requires_grad=False)
        self.stds = nn.Parameter(torch.from_numpy(stds), requires_grad=False)

    def forward(self, x):
        x = self.reshape_input(x)
        x = (x - self.means) / self.stds
        return F.elu(self.norm(self.enc(x)))

    def reshape_input(self, x):
        seq_len, bs, c, h, w = x.shape
        return x.view(seq_len * bs, -1)


class MERTProjector(nn.Module):
    """v13: pre-computed MERT (768-dim, 20fps) → spec_enc dim.
    Input: (sl, bs, 1, 768, 1). Output: (sl*bs, spec_enc).
    """
    def __init__(self, spec_enc, mert_dim=768):
        super().__init__()
        self.n_input_frames = 1
        self.proj = nn.Linear(mert_dim, spec_enc)
        self.norm = nn.LayerNorm(spec_enc)

    def set_stats(self, means, stds): pass

    def forward(self, x):
        sl, bs = x.shape[0], x.shape[1]
        x = x.reshape(sl * bs, -1)
        return F.elu(self.norm(self.proj(x)))


class MERTBiLSTM(nn.Module):
    """v14: 8-frame MERT window → BiLSTM → spec_enc dim.
    Input: (sl, bs, 1, 768, 8). Output: (sl*bs, spec_enc).
    """
    def __init__(self, spec_enc, mert_dim=768, lstm_hidden=256):
        super().__init__()
        self.n_input_frames = 8
        self.lstm = nn.LSTM(mert_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(lstm_hidden * 2, spec_enc)
        self.norm = nn.LayerNorm(spec_enc)

    def set_stats(self, means, stds): pass

    def forward(self, x):
        sl, bs = x.shape[0], x.shape[1]
        n_frames = x.shape[-1]
        x = x.reshape(sl * bs, -1, n_frames).permute(0, 2, 1)  # (sl*bs, n_frames, 768)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=-1)   # (sl*bs, lstm_hidden*2)
        return F.elu(self.norm(self.proj(h)))


class MERTMlpProjector(nn.Module):
    """v15: pre-computed MERT → 2-layer MLP → spec_enc dim.
    Input: (sl, bs, 1, 768, 1). Output: (sl*bs, spec_enc).
    """
    def __init__(self, spec_enc, mert_dim=768, hidden=256):
        super().__init__()
        self.n_input_frames = 1
        self.mlp = nn.Sequential(
            nn.Linear(mert_dim, hidden),
            nn.ELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, spec_enc),
        )
        self.norm = nn.LayerNorm(spec_enc)

    def set_stats(self, means, stds): pass

    def forward(self, x):
        sl, bs = x.shape[0], x.shape[1]
        x = x.reshape(sl * bs, -1)
        return F.elu(self.norm(self.mlp(x)))


class CBEncoder(FBEncoder):
    """Context-based encoder: 40-frame window of 78-bin spectrogram → spec_enc dim.

    Best model in Henkel 2020 (CB_TA). Input shape: (seq_len, bs, 1, 78, 40).
    2D CNN processes the (freq × time) context window.
    spec_enc=32 in their best model config.
    """
    def __init__(self, spec_enc):
        super(CBEncoder, self).__init__(spec_enc)

        self.n_input_frames = 40
        initial = 24

        self.enc = nn.Sequential(
            ConvBlock(1, initial, 3, 1, padding=1),
            ConvBlock(initial, initial, 3, 1, padding=1),
            nn.MaxPool2d(2),

            ConvBlock(initial, initial * 2, 3, 1, padding=1),
            ConvBlock(initial * 2, initial * 2, 3, 1, padding=1),
            nn.MaxPool2d(2),

            ConvBlock(initial * 2, initial * 4, 3, 1, padding=1),
            ConvBlock(initial * 4, initial * 4, 3, 1, padding=1),
            nn.MaxPool2d(2),

            ConvBlock(initial * 4, initial * 4, 3, 1, padding=1),
            ConvBlock(initial * 4, initial * 4, 3, 1, padding=1),
            nn.MaxPool2d(2),

            ConvBlock(initial * 4, initial * 4, 1, 1),
            Flatten(),
            nn.Linear(initial * 4 * 4 * 2, spec_enc)  # 96 * 8 = 768 → spec_enc
        )

        self.means = nn.Parameter(torch.zeros(78).unsqueeze(0).unsqueeze(-1), requires_grad=False)
        self.stds = nn.Parameter(torch.ones(78).unsqueeze(0).unsqueeze(-1), requires_grad=False)

    def set_stats(self, means, stds):
        self.means = nn.Parameter(torch.from_numpy(means).unsqueeze(0).unsqueeze(-1), requires_grad=False)
        self.stds = nn.Parameter(torch.from_numpy(stds).unsqueeze(0).unsqueeze(-1), requires_grad=False)

    def reshape_input(self, x):
        seq_len, bs, c, h, w = x.shape
        return x.view(seq_len * bs, c, h, w)
