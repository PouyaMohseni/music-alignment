"""v9 — Faithful Henkel, Kelz & Widmer ISMIR 2020 (CPJKU) reimplementation.

Key fix vs v8: U-Net operates on the FULL 2D score strip crop (H × W).
The vertical axis encodes pitch (notehead height = pitch position on staff),
so FiLM can learn to activate pitch-specific regions. Collapsing to 1D (v8)
destroyed this signal and caused the model to always predict center.

Architecture:
  Audio: 78-bin log-CQT → 2D CNN (3 blocks, pool freq axis) → LSTM → h_t
  Score: (H × W) strip crop → 2D U-Net, FiLM-conditioned by h_t → heatmap
  Training: Gaussian GT at center of always-centered crop; Dice loss.
  Inference: causal tracking — crop centered at running estimate,
             predict offset from center, update estimate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CPJKUConfig:
    n_bins: int = 78
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_bidirectional: bool = False
    unet_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    h_strip: int = 128       # strip height after resize (power of 2 for U-Net)
    tile_width: int = 512    # score crop width


# ── Audio front-end ────────────────────────────────────────────────────────────

class AudioCNN(nn.Module):
    """78-bin log-CQT (B,1,n_bins,T) → per-frame features (B,T,C).

    Three 2D conv blocks; each pools the frequency axis by 2 while keeping
    the time axis intact (matches Henkel 2020 audio CNN design).
    """
    def __init__(self, n_bins: int = 78, channels: List[int] = (32, 64, 128)):
        super().__init__()
        blocks: list = []
        in_ch = 1
        for ch in channels:
            blocks += [
                nn.Conv2d(in_ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ELU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ELU(inplace=True),
                nn.MaxPool2d((2, 1)),  # pool frequency axis only, keep time
            ]
            in_ch = ch
        blocks.append(nn.AdaptiveAvgPool2d((1, None)))  # collapse remaining freq bins
        self.net = nn.Sequential(*blocks)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)           # (B, C, 1, T)
        return f.squeeze(2).permute(0, 2, 1)  # (B, T, C)


# ── 2-D U-Net with FiLM conditioning ──────────────────────────────────────────

class _FiLM2D(nn.Module):
    def __init__(self, context_dim: int, feat_dim: int):
        super().__init__()
        self.gen = nn.Linear(context_dim, 2 * feat_dim)

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        g, b = self.gen(ctx).chunk(2, dim=-1)          # (B, C) each
        return feat * (1 + g[:, :, None, None]) + b[:, :, None, None]


def _enc2d(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ELU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ELU(inplace=True))


def _dec2d(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ELU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ELU(inplace=True))


class UNet2D(nn.Module):
    """2D U-Net on score strip crop, FiLM-conditioned by LSTM audio context.

    Input:  (B, 1, H, W) score crop    ctx: (B, context_dim)
    Output: (B, 1, H, W) position heatmap in [0, 1]
    """
    def __init__(self, channels: List[int] = (32, 64, 128, 256), context_dim: int = 256):
        super().__init__()
        n = len(channels)

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 1
        for ch in channels:
            self.enc_blocks.append(_enc2d(in_ch, ch))
            in_ch = ch
        for _ in range(n - 1):
            self.pools.append(nn.MaxPool2d(2))

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.film_mods = nn.ModuleList()
        for i in range(n - 2, -1, -1):
            self.up_convs.append(nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2))
            self.dec_blocks.append(_dec2d(channels[i] * 2, channels[i]))
            self.film_mods.append(_FiLM2D(context_dim, channels[i]))

        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.enc_blocks[:-1]):
            x = enc(x)
            skips.append(x)
            x = self.pools[i](x)
        x = self.enc_blocks[-1](x)

        for up, dec, film, skip in zip(
                self.up_convs, self.dec_blocks, self.film_mods, reversed(skips)):
            x = up(x)
            # Pad to match skip size if odd-dimension rounding
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            if dh or dw:
                x = F.pad(x, [0, dw, 0, dh])
            x = dec(torch.cat([x, skip], dim=1))
            x = film(x, ctx)

        return torch.sigmoid(self.head(x))   # (B, 1, H, W)


# ── Full model ─────────────────────────────────────────────────────────────────

class CPJKU(nn.Module):
    """Henkel 2020 audio-conditioned score follower.

    Training:  audio_cqt (B,1,n_bins,T_win) + score_crop (B,1,H,W)
               → pos_map (B,1,H,W), ctx_last (B,lstm_h)
    Inference: audio_context(cqt) → (B,T,lstm_h); then per-frame
               unet(crop, h_t) for causal tracking.
    """
    def __init__(self, cfg: CPJKUConfig | None = None):
        super().__init__()
        self.cfg = cfg or CPJKUConfig()
        ldir = 2 if self.cfg.lstm_bidirectional else 1
        self.audio_cnn = AudioCNN(self.cfg.n_bins, self.cfg.cnn_channels)
        self.lstm = nn.LSTM(
            self.audio_cnn.out_dim, self.cfg.lstm_hidden,
            self.cfg.lstm_layers, batch_first=True,
            bidirectional=self.cfg.lstm_bidirectional)
        ctx_dim = self.cfg.lstm_hidden * ldir
        self.unet = UNet2D(self.cfg.unet_channels, context_dim=ctx_dim)

    def audio_context(self, cqt: torch.Tensor) -> torch.Tensor:
        """cqt: (B,1,n_bins,T) → (B,T,lstm_h)"""
        return self.lstm(self.audio_cnn(cqt))[0]

    def forward(self, audio_cqt: torch.Tensor,
                score_crop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx_seq  = self.audio_context(audio_cqt)   # (B, T, lstm_h)
        ctx_last = ctx_seq[:, -1, :]                # (B, lstm_h)
        pos_map  = self.unet(score_crop, ctx_last)  # (B, 1, H, W)
        return pos_map, ctx_last

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
