"""Full-encoder-replacement experiment: swaps ConditionalUNet's entire
from-scratch 4-stage visual encoder (raw-pixel Conv2d downsampling + skip
connections) for a frozen-DINOv2-derived feature pyramid
(extensions/heads/dinov2_visual_neck.py). Uses plain CBEncoder on the audio
side (NOT MERT) to isolate this as a pure visual-architecture change,
matching this project's "one variable at a time" discipline (same reasoning
as B1a isolating the audio-encoder swap).

This is the one thing in the whole project that changes the network's
actual computational STRUCTURE, not just its training signal (B2-B6/C2) or
a drop-in component with an identical interface (B1a/MERT+X). See
dinov2_visual_neck.py's docstring for the known, accepted resolution-loss
risk being tested here, not assumed.

Three patches, all via module-attribute reassignment (same technique as
every other extensions/hooks/*.py):
  1. audio_conditioned_unet.network.ConditionalUNet -> ConditionalUNetDINOv2Visual
     (subclass; train_model.py's `from audio_conditioned_unet.network import
     ConditionalUNet` picks up the patched class since the reassignment
     happens before that import statement executes).
  2. audio_conditioned_unet.dataset.iterate_dataset -> iterate_dataset_visual
     (forked copy of the stock iterate_dataset that also builds a per-batch-
     slot DINOv2 grid tensor from current_pipeline's file_name and passes it
     to network(..., visual_grid=...) -- the stock network(score=,perf=,
     hidden=) call signature has no hook for this, and batch_size=4 by
     default means different batch slots can be different pieces
     simultaneously, so this can't be done via a single network attribute
     the way per-piece extension state elsewhere in this project is tracked).
  3. Nothing changes on the audio/load_piece side -- plain CBEncoder, plain
     load_piece, exactly like A0.
"""
from __future__ import annotations
import os
from pathlib import Path
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


# ---------------------------------------------------------------------------
# 1. Network subclass
# ---------------------------------------------------------------------------

def _maxpool_out(h: int, w: int) -> tuple[int, int]:
    """Matches nn.MaxPool2d(kernel_size=2, stride=2)'s exact output formula
    (padding=0, dilation=1): floor((size - 2) / 2) + 1. Verified against
    real observed training shapes (393,278)->(196,139)->(98,69)->(49,34)->
    (24,17) before relying on it."""
    return (h - 2) // 2 + 1, (w - 2) // 2 + 1


def _build_dinov2_network(cpjku_network_module, dinov2_root: str):
    from extensions.heads.dinov2_visual_neck import DINOv2VisualNeck

    ConditionalUNet = cpjku_network_module.ConditionalUNet  # original, still needed as base behavior reference
    audio_encoder = cpjku_network_module.audio_encoder
    initialize_weights = cpjku_network_module.initialize_weights

    class ConditionalUNetDINOv2Visual(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.n_encoder_layers = config.get('n_encoder_layers', 4)
            self.n_filters_start = config.get('n_filters_start', 8)
            self.use_lstm = config.get('use_lstm', False)
            self.max_channel = 128

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
                    out_ = min(self.n_filters_start, self.max_channel)
                else:
                    out_ = min(self.n_filters_start * (2 ** (i - 1)), self.max_channel)
                stage_channels.append(out_)

                dec_block = cpjku_network_module.ConditionalUNetBlock(
                    out_, out_, self.rnn_size,
                    film=2 * (self.n_encoder_layers + 1) - i in film_layers,
                    up_in_channels=min(out_ * 2, self.max_channel), up_sample=True, down_sample=False)
                self.decoder.append(dec_block)

            bottleneck_channels = min(self.n_filters_start * (2 ** self.n_encoder_layers), self.max_channel)
            self.bottleneck_block = cpjku_network_module.ConditionalUNetBlock(
                stage_channels[-1], bottleneck_channels, self.rnn_size,
                film=self.n_encoder_layers + 1 in film_layers, down_sample=False)

            self.conv_out = nn.Conv2d(self.n_filters_start, 1, kernel_size=(1, 1))

            self.visual_neck = DINOv2VisualNeck(stage_channels=stage_channels,
                                                bottleneck_channels=stage_channels[-1])

            self.first_execution = True
            self.apply(initialize_weights)

        def forward(self, score, perf, hidden, visual_grid=None):
            if visual_grid is None:
                raise RuntimeError('ConditionalUNetDINOv2Visual requires visual_grid '
                                   '(bs, n_rows, n_cols, 768) -- use iterate_dataset_visual, '
                                   'not the stock iterate_dataset.')

            seq_len, bs, c, h, w = score.shape

            perf = self.perf_encoder(perf)
            if self.use_lstm:
                perf = perf.view(seq_len, bs, -1)
                perf, hidden = self.rnn(perf, hidden)
                perf = perf.view(seq_len * bs, -1)
            else:
                perf = F.elu(self.fc(perf))

            stage_sizes = []
            cur_h, cur_w = h, w
            for _ in range(self.n_encoder_layers):
                stage_sizes.append((cur_h, cur_w))
                cur_h, cur_w = _maxpool_out(cur_h, cur_w)
            bottleneck_size = (cur_h, cur_w)

            # visual_grid: (bs_pieces, n_rows, n_cols, 768), one per BATCH SLOT
            # (constant across that slot's seq_len frames) -- expand to match
            # the (seq_len*bs_pieces) frame-batch dimension used everywhere else.
            grid = visual_grid.unsqueeze(0).expand(seq_len, -1, -1, -1, -1)
            grid = grid.reshape(seq_len * visual_grid.shape[0], *visual_grid.shape[1:])

            residuals, x = self.visual_neck(grid, stage_sizes, bottleneck_size)

            if self.first_execution:
                for r in residuals:
                    print('visual_neck residual', r.shape)

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

    return ConditionalUNetDINOv2Visual


# ---------------------------------------------------------------------------
# 2. iterate_dataset fork (adds visual_grid batching, otherwise verbatim)
# ---------------------------------------------------------------------------

_DINOV2_ROOT = None
_GRID_CACHE: dict[str, torch.Tensor] = {}


def _load_grid(piece_name: str, device) -> torch.Tensor:
    if piece_name not in _GRID_CACHE:
        path = Path(_DINOV2_ROOT) / f'{piece_name}.npy'
        arr = np.load(path).astype(np.float32)   # (n_rows, n_cols, 768)
        _GRID_CACHE[piece_name] = torch.from_numpy(arr)
    return _GRID_CACHE[piece_name].to(device)


def iterate_dataset_visual(network, optimizer, dataset, batch_size, seq_len, train=True, device="cpu",
                           threshold=0.5, average_stats=True, eval_center_of_mass=False,
                           eval_only_onsets=False, clip_grads=None):
    """Verbatim fork of audio_conditioned_unet.dataset.iterate_dataset, with
    ONLY the visual_grid construction/threading added (search 'VISUAL' for
    the three touch points)."""
    from audio_conditioned_unet.dataset import prepare_batch, calculate_batch_stats, summarize_stats
    from audio_conditioned_unet.utils import dice_loss

    dataset.set_random_perfs()
    batch_size = min(batch_size, len(dataset))

    if train:
        network.train()
    else:
        network.eval()

    losses = []
    song_order = [i for i in range(dataset.length)]
    shuffle(song_order)

    progress_bar = tqdm.tqdm(total=len(dataset))

    current_pipeline = []
    for i in range(batch_size):
        if len(song_order) > 0:
            current_pipeline.append(dataset[song_order.pop(0)])

    indices = np.array([0 for i in range(batch_size)])
    lengths = np.array([current_pipeline[i]['inputs']['length'] for i in range(batch_size)])

    use_lstm = hasattr(network, "rnn")
    if use_lstm:
        hidden = (torch.zeros(network.rnn_layers, batch_size, network.rnn_size).to(device),
                  torch.zeros(network.rnn_layers, batch_size, network.rnn_size).to(device))
    else:
        hidden = None

    end_epoch = False
    piece_stats = {}

    while not end_epoch:
        max_seq_length = min(min(lengths - indices), seq_len)

        score_batch, perf_batch, y_batch, onsets = prepare_batch(current_pipeline, indices, max_seq_length, device)

        # VISUAL: one DINOv2 grid per current batch slot, matching prepare_batch's own ordering.
        visual_grid = torch.stack(
            [_load_grid(current_pipeline[idx]['file_name'], device) for idx in range(len(current_pipeline))],
            dim=0)

        bs = score_batch.shape[0] * score_batch.shape[1]

        if bs > 1 or not train:
            with torch.set_grad_enabled(train):
                model_return = network(score=score_batch, perf=perf_batch, hidden=hidden,
                                       visual_grid=visual_grid)   # VISUAL
                pred = model_return['segmentation']
                hidden = model_return['hidden']

            loss = dice_loss(pred, y_batch, smoothing=0.)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if clip_grads is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), clip_grads)
                optimizer.step()

            piece_stats = calculate_batch_stats(pred, y_batch, piece_stats, current_pipeline, onsets,
                                                eval_center_of_mass, eval_only_onsets, threshold)

            if use_lstm:
                hidden = (hidden[0].detach(), hidden[1].detach())

            losses.append(loss.item())

        indices += max_seq_length

        pop_indices = []
        for idx, reset_state in enumerate((indices - lengths) >= 0):
            if reset_state:
                if len(song_order) > 0:
                    current_pipeline[idx] = dataset[song_order.pop(0)]
                    if use_lstm:
                        hidden[0][:, idx] = 0
                        hidden[1][:, idx] = 0
                    indices[idx] = 0
                    lengths[idx] = current_pipeline[idx]['inputs']['length']
                else:
                    pop_indices.append(idx)
                progress_bar.update(1)

        for idx in sorted(pop_indices, reverse=True):
            current_pipeline.pop(idx)
            indices = np.delete(indices, idx)
            lengths = np.delete(lengths, idx)
            if use_lstm:
                h0 = torch.cat((hidden[0][:, :idx], hidden[0][:, idx + 1:]), dim=1)
                h1 = torch.cat((hidden[1][:, :idx], hidden[1][:, idx + 1:]), dim=1)
                hidden = (h0, h1)

        if len(current_pipeline) == 0:
            end_epoch = True

    stats = summarize_stats(piece_stats, average_stats, eval_center_of_mass)
    stats['loss'] = np.mean(losses)
    progress_bar.close()
    return stats


# ---------------------------------------------------------------------------
# 3. Patch entrypoint
# ---------------------------------------------------------------------------

def patch_dinov2_full_encoder(dinov2_root: str):
    global _DINOV2_ROOT
    _DINOV2_ROOT = dinov2_root

    from audio_conditioned_unet import network as cpjku_network
    from audio_conditioned_unet import dataset as cpjku_dataset

    cpjku_network.ConditionalUNet = _build_dinov2_network(cpjku_network, dinov2_root)
    cpjku_dataset.iterate_dataset = iterate_dataset_visual

    print(f'[dinov2_full_encoder_patch] Replaced ConditionalUNet\'s from-scratch visual '
          f'encoder with a DINOv2-tile-grid neck, and iterate_dataset with a visual_grid-aware '
          f'fork (dinov2_root={dinov2_root})', flush=True)
