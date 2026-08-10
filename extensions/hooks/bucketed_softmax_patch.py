"""P1 -- swap the dense soft-Dice objective for a bucketed softmax over x.

The network body is untouched.  We capture `conv_out`'s PRE-SIGMOID output with
a forward hook, marginalise it over height, and train a softmax over x columns
against the GT mask's x-marginal (see extensions/heads/bucketed_softmax.py for
why this is the experiment worth running).

WHY A HOOK RATHER THAN EDITING network.py. ConditionalUNet.forward applies
sigmoid before returning, and the returned tensor is the only handle
iterate_dataset gets.  Recovering logits by inverting the sigmoid is
numerically hopeless once the model saturates -- p=1.0 in float32 maps to inf.
A forward hook on conv_out gets the exact pre-activation values and leaves
third_party untouched, so the submodule stays clean and every other experiment
keeps running unmodified.

DICE_WEIGHT. Defaults to 0.0 -- a PURE reparameterisation, which is what the
MM-Loc-vs-CUNet comparison actually contrasts.  A non-zero value keeps a Dice
term alongside, for the hybrid ablation.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import tqdm
from random import shuffle

from audio_conditioned_unet.dataset import (prepare_batch, calculate_batch_stats,
                                            summarize_stats)
from audio_conditioned_unet.utils import dice_loss

from extensions.heads.bucketed_softmax import bucketed_ce_loss, decode_mask


class LogitCapture:
    """Grabs conv_out's pre-sigmoid output on every forward pass."""

    def __init__(self, network):
        self.logits = None
        if not hasattr(network, 'conv_out'):
            raise RuntimeError('network has no conv_out; cannot capture logits')
        network.conv_out.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.logits = out


def iterate_dataset_bucketed(network, optimizer, dataset, batch_size, seq_len, train=True,
                             device="cpu", threshold=0.5, average_stats=True,
                             eval_center_of_mass=False, eval_only_onsets=False,
                             clip_grads=None, pool='logsumexp', dice_weight=0.0):
    if not hasattr(network, '_ext_logit_capture'):
        network._ext_logit_capture = LogitCapture(network)
    cap = network._ext_logit_capture

    dataset.set_random_perfs()
    batch_size = min(batch_size, len(dataset))
    network.train() if train else network.eval()

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
        score_batch, perf_batch, y_batch, onsets = prepare_batch(
            current_pipeline, indices, max_seq_length, device)
        bs = score_batch.shape[0] * score_batch.shape[1]

        if bs > 1 or not train:
            with torch.set_grad_enabled(train):
                model_return = network(score=score_batch, perf=perf_batch, hidden=hidden)
                pred = model_return['segmentation']
                hidden = model_return['hidden']

            logits = cap.logits                        # (N, 1, H, W), pre-sigmoid
            # y_batch is (sl, bs, c, H, W); flatten to match the conv's (N,1,H,W)
            y_flat = y_batch.view(-1, *y_batch.shape[2:])
            loss, n_valid = bucketed_ce_loss(logits, y_flat, pool=pool)
            if dice_weight > 0.0:
                loss = loss + dice_weight * dice_loss(pred, y_batch, smoothing=0.)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if clip_grads is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), clip_grads)
                optimizer.step()

            # Score through the SAME metric code as every other experiment, by
            # handing it a peak-normalised mask instead of the sigmoid map.
            with torch.no_grad():
                dec = decode_mask(logits.detach(), y_batch.shape[-2], pool=pool)
            piece_stats = calculate_batch_stats(dec, y_batch, piece_stats, current_pipeline,
                                                onsets, eval_center_of_mass, eval_only_onsets,
                                                threshold)

            if use_lstm:
                hidden = (hidden[0].detach(), hidden[1].detach())
            losses.append(loss.item())

        indices += max_seq_length

        pop_indices = []
        for idx, reset_state in enumerate((indices - lengths) >= 0):
            if reset_state:
                progress_bar.update(1)
                if len(song_order) > 0:
                    current_pipeline[idx] = dataset[song_order.pop(0)]
                    lengths[idx] = current_pipeline[idx]['inputs']['length']
                    indices[idx] = 0
                    if use_lstm:      # a new piece must not inherit LSTM state
                        hidden[0][:, idx] = 0
                        hidden[1][:, idx] = 0
                else:
                    pop_indices.append(idx)

        for idx in sorted(pop_indices, reverse=True):
            current_pipeline.pop(idx)
            indices = np.delete(indices, idx)
            lengths = np.delete(lengths, idx)
            if use_lstm:
                hidden = (torch.cat([hidden[0][:, :idx], hidden[0][:, idx + 1:]], dim=1),
                          torch.cat([hidden[1][:, :idx], hidden[1][:, idx + 1:]], dim=1))

        if len(current_pipeline) == 0:
            end_epoch = True

    stats = summarize_stats(piece_stats, average_stats, eval_center_of_mass)
    stats['loss'] = np.mean(losses)
    stats['aux_loss'] = 0.0        # train_model.py logs this key unconditionally

    progress_bar.close()
    return stats


def patch_bucketed_softmax(pool='logsumexp', dice_weight=0.0):
    """Point train_model.py's iterate_dataset at the bucketed version."""
    import functools
    import audio_conditioned_unet.dataset as ds_mod
    import audio_conditioned_unet.train_model as tm

    fn = functools.partial(iterate_dataset_bucketed, pool=pool, dice_weight=dice_weight)
    ds_mod.iterate_dataset = fn
    if hasattr(tm, 'iterate_dataset'):
        tm.iterate_dataset = fn
    print(f'[P1] bucketed-softmax objective ACTIVE: pool={pool} dice_weight={dice_weight}',
          flush=True)
