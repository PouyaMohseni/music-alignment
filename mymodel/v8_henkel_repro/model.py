"""v8 — Faithful Henkel ISMIR 2020 reimplementation.

Architecture mirrors Henkel, Kelz & Widmer 2020 (CPJKU):
  - Audio: 78-bin log-CQT → small 2D CNN → per-frame features → LSTM → h_t
  - Score: strip.png window → 1D U-Net, FiLM-conditioned by h_t → position heatmap
  - Loss: Dice on Gaussian blob at GT strip_x position
  - Inference: causal LSTM + tiled strip → DTW on tile-level confidence,
               sub-tile position from per-tile argmax

This model uses RAW audio (audio.wav) and RAW score images (strip.png) —
NOT MERT or ViT embeddings — so it can reproduce Henkel's 85.2% @0.5s.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HenkelReproConfig:
    # Audio front-end (CQT CNN)
    n_bins: int = 78
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    # LSTM
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_bidirectional: bool = False   # causal for online following
    # Score U-Net
    unet_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    tile_width: int = 512              # strip windows resized to this for U-Net input


# ── Audio front-end ──────────────────────────────────────────────────────────

class AudioCNN(nn.Module):
    """2D CNN over log-CQT spectrogram → per-frame feature vectors.

    Input : (B, 1, n_bins, T)
    Output: (B, T, C)  where C = cnn_channels[-1]
    """
    def __init__(self, n_bins: int = 78, channels: List[int] = (32, 64, 128)):
        super().__init__()
        layers: list = []
        in_ch = 1
        for ch in channels:
            layers += [
                nn.Conv2d(in_ch, ch, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(ch), nn.ELU(inplace=True)]
            in_ch = ch
        # Pool over the entire frequency axis, keep time axis intact
        layers.append(nn.AdaptiveAvgPool2d((1, None)))
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)           # (B, C, 1, T)
        f = f.squeeze(2)          # (B, C, T)
        return f.permute(0, 2, 1) # (B, T, C)


# ── 1-D U-Net with FiLM conditioning ─────────────────────────────────────────

class _FiLM(nn.Module):
    def __init__(self, context_dim: int, feat_dim: int):
        super().__init__()
        self.gen = nn.Linear(context_dim, 2 * feat_dim)

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, W)   ctx: (B, context_dim)
        g, b = self.gen(ctx).chunk(2, dim=-1)   # (B, C) each
        return feat * (1.0 + g.unsqueeze(-1)) + b.unsqueeze(-1)


def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, 7, padding=3), nn.BatchNorm1d(out_ch), nn.ELU(inplace=True),
        nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ELU(inplace=True))


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ELU(inplace=True),
        nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ELU(inplace=True))


class _EncOut(NamedTuple):
    skips: list       # list of (B, ch_i, W_i) tensors, shallow → deep
    bottom: torch.Tensor  # (B, ch_last, W_last)


class UNet1D(nn.Module):
    """1D U-Net on grayscale score strip windows, FiLM-conditioned by LSTM context.

    Supports encode/decode split for efficient inference:
      enc = model.encode(strip_tiles)   # run once per piece
      pos_map = model.decode(enc, ctx)  # run T times (one per audio frame)
    """
    def __init__(self, channels: List[int] = (32, 64, 128, 256), context_dim: int = 256):
        super().__init__()
        n = len(channels)

        self.enc_blocks = nn.ModuleList()
        self.pools       = nn.ModuleList()
        in_ch = 1
        for ch in channels:
            self.enc_blocks.append(_enc_block(in_ch, ch))
            in_ch = ch
        for _ in range(n - 1):
            self.pools.append(nn.MaxPool1d(2))

        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.film_mods  = nn.ModuleList()
        for i in range(n - 2, -1, -1):
            ch_up   = channels[i + 1]
            ch_skip = channels[i]
            self.up_convs.append(nn.ConvTranspose1d(ch_up, ch_skip, 2, stride=2))
            self.dec_blocks.append(_dec_block(ch_skip * 2, ch_skip))
            self.film_mods.append(_FiLM(context_dim, ch_skip))

        self.head = nn.Conv1d(channels[0], 1, 1)

    def encode(self, x: torch.Tensor) -> _EncOut:
        """x: (B, 1, W) → encoder features for later FiLM-conditioned decode."""
        skips = []
        for i, enc in enumerate(self.enc_blocks[:-1]):
            x = enc(x)
            skips.append(x)
            x = self.pools[i](x)
        bottom = self.enc_blocks[-1](x)
        return _EncOut(skips=skips, bottom=bottom)

    def decode(self, enc: _EncOut, ctx: torch.Tensor) -> torch.Tensor:
        """enc: from encode()   ctx: (B, context_dim) → (B, 1, W) position heatmap."""
        x = enc.bottom
        for up, dec, film, skip in zip(
                self.up_convs, self.dec_blocks, self.film_mods,
                reversed(enc.skips)):
            x = up(x)
            # Pad/crop if W mismatch from odd-length strip (should be rare)
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = dec(torch.cat([x, skip], dim=1))
            x = film(x, ctx)
        return torch.sigmoid(self.head(x))   # (B, 1, W)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), ctx)


# ── Full model ────────────────────────────────────────────────────────────────

class HenkelRepro(nn.Module):
    """Faithful Henkel 2020 reimplementation.

    Training forward:
      audio_cqt  (B, 1, n_bins, T_win) — CQT of audio window
      strip_win  (B, 1, tile_width)     — score strip window, resized
      → pos_map  (B, 1, tile_width)     — predicted position heatmap
        ctx      (B, lstm_h)            — LSTM context (last frame)

    Inference:
      1. audio_feats(cqt) → (T, lstm_h) context vectors
      2. encode_tiles(tiles) → cached encoder features  (once per piece)
      3. For each frame t, decode(enc, ctx_t) → (N, 1, W) → argmax → strip_x
    """

    def __init__(self, cfg: HenkelReproConfig | None = None):
        super().__init__()
        self.cfg = cfg or HenkelReproConfig()
        lh = self.cfg.lstm_hidden
        ldir = 2 if self.cfg.lstm_bidirectional else 1

        self.audio_cnn = AudioCNN(self.cfg.n_bins, self.cfg.cnn_channels)
        self.lstm = nn.LSTM(
            self.audio_cnn.out_dim, lh, self.cfg.lstm_layers,
            batch_first=True, bidirectional=self.cfg.lstm_bidirectional)

        ctx_dim = lh * ldir
        self.unet = UNet1D(self.cfg.unet_channels, context_dim=ctx_dim)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def audio_context(self, cqt: torch.Tensor) -> torch.Tensor:
        """cqt: (B, 1, n_bins, T) → (B, T, lstm_h) context sequence."""
        feat = self.audio_cnn(cqt)    # (B, T, C_a)
        out, _ = self.lstm(feat)      # (B, T, lstm_h)
        return out

    def forward(self, audio_cqt: torch.Tensor,
                strip_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward. Returns (pos_map (B,1,W), ctx_last (B,lstm_h))."""
        ctx_seq = self.audio_context(audio_cqt)  # (B, T, lstm_h)
        ctx_last = ctx_seq[:, -1, :]              # last frame context: (B, lstm_h)
        pos_map = self.unet(strip_win, ctx_last)  # (B, 1, W)
        return pos_map, ctx_last
