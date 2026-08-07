"""Gated-residual DINOv2 hybrid: unlike V-DINOv2 (dinov2_full_encoder_patch.py,
6.9% pct@0.5s -- a full REPLACEMENT of CB_TA's from-scratch conv encoder,
catastrophic due to the DINOv2 tile grid's much coarser native resolution),
this KEEPS the original from-scratch encoder as the primary path and adds
the DINOv2 tile-grid features only as a zero-initialized additive residual
at each stage -- the same "start as identity, let training decide how much
to trust the new signal" idea that rescued gated FiLM (82.9%) over
full-strength cross-attention/spatial FiLM (71.1%/44.3%), applied here to
the visual BACKBONE instead of the conditioning mechanism.

At initialization every gate is exactly 0, so the network is IDENTICAL to
plain CB_TA (A0) -- training can only improve on that baseline by actually
learning to extract something useful from the coarse DINOv2 tokens, never
regress below it the way full replacement did.

Uses plain CBEncoder audio (NOT MERT), matching V-DINOv2's isolation
discipline -- this tests a pure visual-architecture idea.

Gate convention: `combined = stock + gate * (dinov2_adapted - stock)`, one
learnable per-channel gate PARAMETER per stage (not conditioned on
anything -- this is a residual-pathway blend, not a FiLM site, so a plain
zero-initialized nn.Parameter per stage is the right analogue of ResNet/DiT
zero-init residual gating, not an AdaLN-Zero-style predicted gate).

Reuses iterate_dataset_visual + the DINOv2 grid loader/cache from
extensions/hooks/dinov2_full_encoder_patch.py unchanged for visual_grid
batching.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _maxpool_out(h: int, w: int) -> tuple[int, int]:
    """Matches nn.MaxPool2d(kernel_size=2, stride=2)'s exact output formula
    (padding=0, dilation=1): floor((size - 2) / 2) + 1."""
    return (h - 2) // 2 + 1, (w - 2) // 2 + 1


def _build_gated_dinov2_residual_network(cpjku_network_module, dinov2_root: str):
    from extensions.heads.dinov2_visual_neck import DINOv2VisualNeck

    audio_encoder = cpjku_network_module.audio_encoder
    initialize_weights = cpjku_network_module.initialize_weights
    ConditionalUNetBlock = cpjku_network_module.ConditionalUNetBlock

    class ConditionalUNetGatedDINOv2Residual(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.n_encoder_layers = config.get('n_encoder_layers', 4)
            self.n_filters_start = config.get('n_filters_start', 8)
            self.use_lstm = config.get('use_lstm', False)
            self.max_channel = 128

            self.encoder = nn.ModuleList()
            self.decoder = nn.ModuleList()
            self.rnn_size = config.get('rnn_size', 512)
            self.rnn_layers = config.get('rnn_layer', 1)
            self.spec_enc = config.get('spec_enc', 512)

            self.perf_encoder = getattr(audio_encoder, config['audio_encoder'])(self.spec_enc)

            if self.use_lstm:
                self.rnn = nn.LSTM(self.spec_enc, hidden_size=self.rnn_size,
                                   num_layers=self.rnn_layers, batch_first=False)
            else:
                self.fc = nn.Linear(self.spec_enc, self.rnn_size)

            film_layers = config['film_layers']
            stage_channels = []
            for i in range(1, self.n_encoder_layers + 1):
                if i == 1:
                    in_ = 1
                    out_ = min(self.n_filters_start, self.max_channel)
                else:
                    in_ = min(self.n_filters_start * (2 ** (i - 2)), self.max_channel)
                    out_ = min(self.n_filters_start * (2 ** (i - 1)), self.max_channel)
                stage_channels.append(out_)

                enc_block = ConditionalUNetBlock(in_, out_, self.rnn_size, film=i in film_layers)
                dec_block = ConditionalUNetBlock(
                    out_, out_, self.rnn_size,
                    film=2 * (self.n_encoder_layers + 1) - i in film_layers,
                    up_in_channels=min(out_ * 2, self.max_channel), up_sample=True, down_sample=False)
                self.encoder.append(enc_block)
                self.decoder.append(dec_block)

            bottleneck_channels = min(self.n_filters_start * (2 ** self.n_encoder_layers), self.max_channel)
            self.bottleneck_block = ConditionalUNetBlock(
                stage_channels[-1], bottleneck_channels, self.rnn_size,
                film=self.n_encoder_layers + 1 in film_layers, down_sample=False)

            self.conv_out = nn.Conv2d(self.n_filters_start, 1, kernel_size=(1, 1))

            self.visual_neck = DINOv2VisualNeck(stage_channels=stage_channels,
                                                bottleneck_channels=stage_channels[-1])

            # One zero-initialized per-channel gate per stage residual + one
            # for the bottleneck input -- plain nn.Parameter, not predicted
            # from any conditioning signal (this blends two VISUAL feature
            # pathways, not an audio-conditioned FiLM site).
            self.stage_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(1, c, 1, 1)) for c in stage_channels])
            # Gates the tensor GOING INTO bottleneck_block (channel count
            # stage_channels[-1]), not bottleneck_block's own output
            # (bottleneck_channels) -- x at the gating point still has
            # stage_channels[-1] channels; bottleneck_block is what expands
            # to bottleneck_channels.
            self.bottleneck_gate = nn.Parameter(torch.zeros(1, stage_channels[-1], 1, 1))
            for g in self.stage_gates:
                g._gated_film_zero_init = True   # tag only for documentation parity; these are
            self.bottleneck_gate._gated_film_zero_init = True   # Parameters, never touched by
                                                                  # self.apply(initialize_weights)
                                                                  # (that only visits nn.Module
                                                                  # instances with weight/bias
                                                                  # attributes, not raw Parameters).

            self.first_execution = True
            self.apply(initialize_weights)

        def forward(self, score, perf, hidden, visual_grid=None):
            if visual_grid is None:
                raise RuntimeError('ConditionalUNetGatedDINOv2Residual requires visual_grid '
                                   '(bs, n_rows, n_cols, 768) -- use iterate_dataset_visual, '
                                   'not the stock iterate_dataset.')

            x = score
            seq_len, bs, c, h, w = score.shape
            x = x.view(seq_len * bs, c, h, w)

            perf = self.perf_encoder(perf)
            if self.use_lstm:
                perf = perf.view(seq_len, bs, -1)
                perf, hidden = self.rnn(perf, hidden)
                perf = perf.view(seq_len * bs, -1)
            else:
                perf = F.elu(self.fc(perf))

            stock_residuals = []
            for i in range(self.n_encoder_layers):
                res, x = self.encoder[i](x, perf)
                stock_residuals.append(res)

            stage_sizes = [r.shape[-2:] for r in stock_residuals]
            bottleneck_size = x.shape[-2:]

            grid = visual_grid.unsqueeze(0).expand(seq_len, -1, -1, -1, -1)
            grid = grid.reshape(seq_len * visual_grid.shape[0], *visual_grid.shape[1:])
            dinov2_residuals, dinov2_bottleneck = self.visual_neck(grid, stage_sizes, bottleneck_size)

            residuals = [stock_residuals[i] + self.stage_gates[i] * (dinov2_residuals[i] - stock_residuals[i])
                        for i in range(self.n_encoder_layers)]
            x = x + self.bottleneck_gate * (dinov2_bottleneck - x)

            if self.first_execution:
                for r in residuals:
                    print('gated residual', r.shape)
                print('gated bottleneck input', x.shape)

            x = self.bottleneck_block(x, perf)
            if self.first_execution:
                print('bottleneck', x.shape)

            for i in range(self.n_encoder_layers)[::-1]:
                x = self.decoder[i](x, perf, residuals[i])
                if self.first_execution:
                    print('up', x.shape)

            x = self.conv_out(x)
            if self.first_execution:
                print('out', x.shape)
                self.first_execution = False

            x = torch.sigmoid(x)
            return {'segmentation': x, 'hidden': hidden}

    return ConditionalUNetGatedDINOv2Residual


def patch_gated_dinov2_residual(dinov2_root: str):
    from extensions.hooks import dinov2_full_encoder_patch as _dfp
    _dfp._DINOV2_ROOT = dinov2_root

    from audio_conditioned_unet import network as cpjku_network
    from audio_conditioned_unet import dataset as cpjku_dataset

    cpjku_network.ConditionalUNet = _build_gated_dinov2_residual_network(cpjku_network, dinov2_root)
    cpjku_dataset.iterate_dataset = _dfp.iterate_dataset_visual

    print(f'[gated_dinov2_residual_patch] Kept CB_TA\'s from-scratch encoder as the primary '
          f'path, added DINOv2 tile-grid features as a zero-initialized gated residual '
          f'(starts identical to plain CB_TA; dinov2_root={dinov2_root})', flush=True)
