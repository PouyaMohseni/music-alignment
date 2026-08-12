"""A4+A3 -- boundary-oriented output with an explicit coarse-staff decision.

Replaces the dense soft-Dice objective with:
    boundary_loss  anchor CE (uniform over the span) + edge-offset L1 + 1-D IoU
    staff_loss     soft CE over coarse y bands
and decodes position as midpoint(anchor - d_left, anchor + d_right) at the row
the staff head selects.

WHY THESE TWO TOGETHER. They are the x and y halves of one prediction, and each
fixes a distinct measured failure:

  * A4 (x). Every output we have built predicts a CENTER -- the soft-Dice
    argmax, the bucketed-softmax argmax, CYOLO's box centre. BAM-DETR
    (ECCV 2024) shows centre prediction is ill-posed ("centre misalignment from
    the inherent ambiguity of moment centres") and that predicting an interior
    anchor plus edge offsets fixes it, with the largest gains at TIGHT
    tolerances -- where we are weakest (32.9 / 34.6 at 0.05 / 0.1 s against
    56.6 at 0.5 s).
  * A3 (y). Staff assignment carries an enormous share of the metric and has
    never been supervised directly: calculate_batch_stats uses the prediction's
    y to pick a staff and then unrolls x against THAT staff's offset, so a
    model with perfect x and the wrong staff scores near zero. That is
    literally what happened to P1's first run (10.6 on room). cyolo_sb's +4.9
    over cyolo comes from adding exactly this kind of coarse structure.

New parameters are created lazily and registered via
optimizer.add_param_group, following extensions/losses/b2_callback.py: the
optimizer is built by train_model.py before iterate_dataset ever runs, and the
head's channel count is only known once a real decoder feature is captured.
add_module also puts them in the state_dict so checkpoints round-trip.
"""
from __future__ import annotations

import numpy as np
import torch
import tqdm
from random import shuffle

from extensions.heads.boundary_head import (BoundaryHead, boundary_loss,
                                            decode_mask as boundary_decode_mask)
from extensions.heads.staff_coarse_head import StaffCoarseHead, staff_loss

# audio_conditioned_unet is imported lazily inside the loop: importing it at
# module level dies before anything has made the cpjku package importable
# (this cost job 551057 an A100 allocation).


def _staff_rows_from_logits(staff_logits: torch.Tensor, height: int) -> torch.Tensor:
    """(N, n_bins) -> (N,) row index at the centre of the argmax band."""
    n_bins = staff_logits.shape[1]
    b = staff_logits.argmax(dim=1).float()
    return ((b + 0.5) * height / n_bins).clamp(0, height - 1)


def iterate_dataset_boundary(network, optimizer, dataset, batch_size, seq_len, train=True,
                             device="cpu", threshold=0.5, average_stats=True,
                             eval_center_of_mass=False, eval_only_onsets=False,
                             clip_grads=None, decoder_stage=6, n_staff_bins=16,
                             w_staff=1.0, dice_weight=0.0):
    from audio_conditioned_unet.dataset import (prepare_batch, calculate_batch_stats,
                                                summarize_stats)
    from audio_conditioned_unet.utils import dice_loss
    from extensions.hooks.film_feature_extractor import (FeatureCapture,
                                                         decoder_index_for_stage)

    if not hasattr(network, '_ext_bnd_capture'):
        idx = decoder_index_for_stage(network.n_encoder_layers, decoder_stage)
        network._ext_bnd_capture = FeatureCapture(network, idx)
    cap = network._ext_bnd_capture

    dataset.set_random_perfs()
    batch_size = min(batch_size, len(dataset))
    network.train() if train else network.eval()

    losses, parts_acc = [], {'anchor': [], 'off': [], 'iou': [], 'staff': []}
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

            feat = cap.feature
            if feat is None:
                raise RuntimeError('decoder feature hook never fired')

            # lazily build the heads: channel count is only known now
            if not hasattr(network, '_ext_bnd_head'):
                bh = BoundaryHead(in_ch=feat.shape[1]).to(feat.device)
                sh = StaffCoarseHead(in_ch=feat.shape[1], n_bins=n_staff_bins).to(feat.device)
                network.add_module('_ext_bnd_head', bh)
                network.add_module('_ext_staff_head', sh)
                optimizer.add_param_group({'params': bh.parameters()})
                optimizer.add_param_group({'params': sh.parameters()})
                print(f'[A4] BoundaryHead + StaffCoarseHead created on {feat.shape[1]}ch '
                      f'decoder stage {decoder_stage}; registered with optimizer', flush=True)
            bh, sh = network._ext_bnd_head, network._ext_staff_head

            H, W = y_batch.shape[-2], y_batch.shape[-1]
            y_flat = y_batch.view(-1, *y_batch.shape[2:])

            a_log, dl, dr = bh(feat)
            s_log = sh(feat)

            # The heads run on a DOWNSAMPLED decoder stage, so their column axis
            # is shorter than the mask's. Resize the GT to the head's width
            # rather than upsampling the prediction: interpolating logits would
            # invent boundary precision the features do not have.
            if a_log.shape[1] != W:
                y_small = torch.nn.functional.adaptive_avg_pool2d(
                    y_flat, (y_flat.shape[-2], a_log.shape[1]))
            else:
                y_small = y_flat

            l_bnd, parts = boundary_loss(a_log, dl, dr, y_small)
            l_staff, _ = staff_loss(s_log, y_flat, n_staff_bins)
            loss = l_bnd + w_staff * l_staff
            if dice_weight > 0.0:
                loss = loss + dice_weight * dice_loss(pred, y_batch, smoothing=0.)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if clip_grads is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), clip_grads)
                optimizer.step()

            with torch.no_grad():
                rows = _staff_rows_from_logits(s_log.detach(), H)
                # scale the decoded column back to the mask's width
                scale = W / a_log.shape[1]
                dec = boundary_decode_mask(a_log.detach(), dl.detach() * scale,
                                           dr.detach() * scale, H, staff_row=rows)
                if dec.shape[-1] != W:
                    dec = torch.nn.functional.interpolate(dec, size=(H, W), mode='nearest')
            piece_stats = calculate_batch_stats(dec, y_batch, piece_stats, current_pipeline,
                                                onsets, eval_center_of_mass, eval_only_onsets,
                                                threshold)

            if use_lstm:
                hidden = (hidden[0].detach(), hidden[1].detach())
            losses.append(loss.item())
            for k in ('anchor', 'off', 'iou'):
                parts_acc[k].append(parts[k])
            parts_acc['staff'].append(float(l_staff))

        indices += max_seq_length

        pop_indices = []
        for idx_, reset_state in enumerate((indices - lengths) >= 0):
            if reset_state:
                progress_bar.update(1)
                if len(song_order) > 0:
                    current_pipeline[idx_] = dataset[song_order.pop(0)]
                    lengths[idx_] = current_pipeline[idx_]['inputs']['length']
                    indices[idx_] = 0
                    if use_lstm:
                        hidden[0][:, idx_] = 0
                        hidden[1][:, idx_] = 0
                else:
                    pop_indices.append(idx_)

        for idx_ in sorted(pop_indices, reverse=True):
            current_pipeline.pop(idx_)
            indices = np.delete(indices, idx_)
            lengths = np.delete(lengths, idx_)
            if use_lstm:
                hidden = (torch.cat([hidden[0][:, :idx_], hidden[0][:, idx_ + 1:]], dim=1),
                          torch.cat([hidden[1][:, :idx_], hidden[1][:, idx_ + 1:]], dim=1))

        if len(current_pipeline) == 0:
            end_epoch = True

    stats = summarize_stats(piece_stats, average_stats, eval_center_of_mass)
    stats['loss'] = float(np.mean(losses))
    stats['aux_loss'] = float(np.mean(parts_acc['staff'])) if parts_acc['staff'] else 0.0
    progress_bar.close()
    print('[A4] anchor=%.4f off=%.4f iou=%.4f staff=%.4f'
          % tuple(float(np.mean(parts_acc[k])) if parts_acc[k] else 0.0
                  for k in ('anchor', 'off', 'iou', 'staff')), flush=True)
    return stats


def patch_boundary(decoder_stage=6, n_staff_bins=16, w_staff=1.0, dice_weight=0.0):
    import functools

    import audio_conditioned_unet.dataset as ds_mod
    import audio_conditioned_unet.train_model as tm

    fn = functools.partial(iterate_dataset_boundary, decoder_stage=decoder_stage,
                           n_staff_bins=n_staff_bins, w_staff=w_staff,
                           dice_weight=dice_weight)
    ds_mod.iterate_dataset = fn
    if hasattr(tm, 'iterate_dataset'):
        tm.iterate_dataset = fn
    print(f'[A4] boundary+staff objective ACTIVE (stage={decoder_stage}, '
          f'bins={n_staff_bins}, w_staff={w_staff}, dice={dice_weight})', flush=True)
