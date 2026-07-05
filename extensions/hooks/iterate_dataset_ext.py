"""Extensible fork of audio_conditioned_unet.dataset.iterate_dataset that
adds a pluggable auxiliary loss, reused by B2 (pitch aux), B3 (INR
sub-pixel), B4 (temporal consistency), and B5 (dense contrastive) instead of
each maintaining its own near-duplicate copy of the training loop.

Only the loss computation differs from the base version (dice + optional
weighted auxiliary term); batching, piece-pipeline management, hidden-state
handling, and stats are copied verbatim from dataset.py's iterate_dataset so
behavior matches A0/B1a exactly when aux_loss_fn=None.

aux_loss_fn signature:
  (pred, y_batch, rnn_out, decoder_feature, seq_len, bs, network, optimizer) -> (loss, log_dict)
  pred:            (seq_len*bs, 1, H, W) raw sigmoid segmentation, same as dice_loss's input
  y_batch:         (seq_len, bs, 1, H, W) GT heatmap
  rnn_out:         (seq_len*bs, rnn_size) LSTM output (the FiLM-conditioning vector, i.e. the
                   ALIGNED representation -- NOT a frozen/detached input; see REDESIGN.md 9.1
                   for why sampling from anything else silently trains a dead-end side branch)
  decoder_feature: (seq_len*bs, C, H', W') post-FiLM feature map from a chosen decoder stage,
                   or None if no FeatureCapture was supplied
  seq_len, bs:     ints, actual (max_seq_length, batch_size) for this chunk
  network, optimizer: passed through so a callback needing its OWN trainable
                   module (e.g. B5's audio_proj) can lazily create it on first
                   call and register it via optimizer.add_param_group(...) --
                   train_model.py builds the optimizer from network.parameters()
                   BEFORE iterate_dataset is ever called, so any module added
                   to `network` afterward needs this explicit registration or
                   its parameters silently never receive a step.
  pitch_batch:     (seq_len, bs, 88) active-MIDI-pitch multi-hot target, or None
                   if need_pitch_roll=False. Requires extensions/hooks/pitch_patch.py's
                   patch_pitch_pipeline() to have been called (B2 only -- the only
                   extension needing data the base pipeline doesn't already track).
"""
from random import shuffle

import numpy as np
import torch
import tqdm

from audio_conditioned_unet.dataset import prepare_batch, calculate_batch_stats, summarize_stats
from audio_conditioned_unet.utils import dice_loss


class RNNOutCapture:
    """Captures network.rnn's output (the FiLM-conditioning vector) each
    forward call -- this is the ALIGNED representation the network actually
    uses, not a frozen precomputed input."""

    def __init__(self, network):
        self.output = None
        self._handle = network.rnn.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        rnn_out, _ = output
        self.output = rnn_out.reshape(-1, rnn_out.shape[-1])   # (seq_len*bs, rnn_size)

    def remove(self):
        self._handle.remove()


def _prepare_pitch_batch(current_pipeline, indices, max_seq_length, device):
    """Mirrors prepare_batch's slicing exactly, for the extra 'pitch_roll'
    target field that base prepare_batch doesn't know about (B2 only)."""
    import torch
    pitch_batch = []
    for idx, data in enumerate(current_pipeline):
        pr = data['targets']['pitch_roll']
        start = indices[idx]
        pitch_batch.append(np.expand_dims(pr[start:start + max_seq_length], 1))
    return torch.from_numpy(np.concatenate(pitch_batch, axis=1)).to(device)   # (seq_len, bs, 88)


def iterate_dataset_ext(network, optimizer, dataset, batch_size, seq_len, train=True, device="cpu",
                        threshold=0.5, average_stats=True, eval_center_of_mass=False,
                        eval_only_onsets=False, clip_grads=None,
                        aux_loss_fn=None, aux_loss_weight=1.0,
                        need_rnn_capture=False, decoder_feature_stage=None,
                        need_pitch_roll=False):
    """need_rnn_capture / decoder_feature_stage: hooks are attached to
    `network` lazily on first call (network doesn't exist yet when this
    function is bound via functools.partial in the wrapper script, before
    train_model.py constructs it) and cached as network._ext_* attributes."""
    rnn_capture = None
    feature_capture = None
    if need_rnn_capture:
        if not hasattr(network, '_ext_rnn_capture'):
            network._ext_rnn_capture = RNNOutCapture(network)
        rnn_capture = network._ext_rnn_capture
    if decoder_feature_stage is not None:
        if not hasattr(network, '_ext_feature_capture'):
            from extensions.hooks.film_feature_extractor import FeatureCapture, decoder_index_for_stage
            idx = decoder_index_for_stage(network.n_encoder_layers, decoder_feature_stage)
            network._ext_feature_capture = FeatureCapture(network, idx)
        feature_capture = network._ext_feature_capture

    dataset.set_random_perfs()
    batch_size = min(batch_size, len(dataset))

    if train:
        network.train()
    else:
        network.eval()

    losses = []
    aux_losses = []

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

        bs = score_batch.shape[0] * score_batch.shape[1]

        if bs > 1 or not train:
            with torch.set_grad_enabled(train):
                model_return = network(score=score_batch, perf=perf_batch, hidden=hidden)
                pred = model_return['segmentation']
                hidden = model_return['hidden']

            loss = dice_loss(pred, y_batch, smoothing=0.)

            if aux_loss_fn is not None:
                rnn_out = rnn_capture.output if rnn_capture is not None else None
                dec_feat = feature_capture.feature if feature_capture is not None else None
                pitch_batch = (_prepare_pitch_batch(current_pipeline, indices, max_seq_length, device)
                              if need_pitch_roll else None)
                aux_loss, _ = aux_loss_fn(pred, y_batch, rnn_out, dec_feat,
                                          score_batch.shape[0], score_batch.shape[1],
                                          network, optimizer, pitch_batch)
                loss = loss + aux_loss_weight * aux_loss
                aux_losses.append(aux_loss.item())

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
    stats['aux_loss'] = np.mean(aux_losses) if aux_losses else 0.0

    progress_bar.close()
    return stats
