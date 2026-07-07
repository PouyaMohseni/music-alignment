"""C5 -- Test-Time Per-Piece Calibration.

Unlike open-vocabulary speech alignment, the score (and therefore the piece
identity) is known in advance for score-following. This exploits that: run
a handful of gradient steps against the FIRST FEW SECONDS of a piece's own
audio+score (where the ground-truth heatmap is used purely as a per-piece
"enrollment" signal, not smuggled into the reported metric), then score
ONLY the remainder of the piece with the now-calibrated weights.

Only the final 1x1 conv_out layer is unfrozen for calibration -- cheap
(small parameter count, fast per-piece optimization) and it's the exact
layer that turns FiLM-conditioned features into the sigmoid heatmap, so a
few steps of recalibration there can correct for piece-specific quirks
(instrument voicing, rendering tempo, etc.) without touching the shared
CNN/RNN feature extractors.

Weights are restored after each piece so calibration never leaks across
pieces (this module deep-copies conv_out's original state and reloads it
before returning).
"""
from __future__ import annotations
import copy

import numpy as np
import torch
import torch.nn.functional as F


def _build_gt_mask(score_shape, true_pos_yxh, gt_width):
    """Matches ScoreAudioDataset.__getitem__'s (unaugmented) GT construction:
    a box of `gt_width` (x) x `height` (y, adaptive per-staff) around the
    true position, zero elsewhere."""
    y, x, height = int(true_pos_yxh[0]), int(true_pos_yxh[1]), int(true_pos_yxh[2])
    mask = np.zeros(score_shape, dtype=np.float32)
    mask[y - height // 2:y + height // 2, x - gt_width // 2:x + gt_width // 2] = 1
    return mask


def _make_clip(spec, i, n_frames):
    clip = spec[:, max(0, i - n_frames + 1):i + 1]
    if clip.shape[-1] < n_frames:
        clip = np.pad(clip, ((0, 0), (n_frames - clip.shape[-1], 0)))
    return clip


def dice_loss(inputs, targets, smoothing=1., eps=1e-8):
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smoothing) /
                ((iflat ** 2).sum() + (tflat ** 2).sum() + smoothing + eps))


def calibrate_and_infer_piece(network, score, spec, interpol_fnc, interpol_c2o,
                              add_per_staff, onsets_set,
                              pad, gt_width, n_frames, device,
                              calib_seconds=8.0, fps=20, num_steps=15, lr=1e-3,
                              seq_len=8, threshold=0.5, max_eval_seconds=None):
    """Runs test-time calibration on the first `calib_seconds` of one piece,
    then performs normal chunked inference on the remainder.

    Returns onset_diffs (list[int], in FRAMES -- caller divides by fps to get
    seconds, matching eval_official.py's convention) computed ONLY over the
    post-calibration (eval) segment.

    max_eval_seconds: optional cap on how much of the eval segment to score
    (for cheap debugging/smoke-testing on CPU) -- None (default) scores the
    whole remainder of the piece, as the real eval script always should.
    """
    T_total = spec.shape[-1]
    calib_frames = int(round(calib_seconds * fps))
    calib_end = min(pad + calib_frames, T_total)
    if max_eval_seconds is not None:
        T_total = min(T_total, calib_end + int(round(max_eval_seconds * fps)))

    score_t = torch.from_numpy(score[np.newaxis, np.newaxis, np.newaxis]).to(device)

    # --- snapshot conv_out (the only layer calibration touches) ---
    conv_out_backup = copy.deepcopy(network.conv_out.state_dict())

    for p in network.parameters():
        p.requires_grad_(False)
    for p in network.conv_out.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(network.conv_out.parameters(), lr=lr)

    calib_range = list(range(pad, calib_end))
    hidden = None

    if calib_range:
        clips = np.array([_make_clip(spec, i, n_frames) for i in calib_range])
        perf_t = torch.from_numpy(clips[:, np.newaxis, np.newaxis]).to(device)
        gt = np.stack([
            _build_gt_mask(score.shape, interpol_fnc(i - pad), gt_width)
            for i in calib_range
        ])
        gt_t = torch.from_numpy(gt[:, np.newaxis]).to(device)
        score_batch = score_t.expand(len(calib_range), -1, -1, -1, -1)

        network.train()
        first_loss = None
        last_loss = None
        for step in range(num_steps):
            opt.zero_grad()
            out = network(score=score_batch, perf=perf_t, hidden=None)
            pred = out['segmentation']
            loss = dice_loss(pred, gt_t)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            if first_loss is None:
                first_loss = last_loss
        network.eval()

        # one more forward (no grad, calibrated weights) to carry `hidden`
        # continuity into the eval segment.
        with torch.no_grad():
            out = network(score=score_batch, perf=perf_t, hidden=None)
            hidden = out.get('hidden')
        calib_initial_loss, calib_final_loss = first_loss, last_loss
    else:
        calib_initial_loss = calib_final_loss = None

    # --- normal chunked inference over the EVAL segment only ---
    from audio_conditioned_unet.utils import center_of_mass

    onset_diffs = []
    t = calib_end
    network.eval()
    with torch.no_grad():
        while t < T_total:
            end = min(t + seq_len, T_total)
            frame_range = list(range(t, end))
            sl = len(frame_range)

            clips = np.array([_make_clip(spec, i, n_frames) for i in frame_range])
            perf_t = torch.from_numpy(clips[:, np.newaxis, np.newaxis]).to(device)
            score_batch = score_t.expand(sl, -1, -1, -1, -1)

            out = network(score=score_batch, perf=perf_t, hidden=hidden)
            pred = out['segmentation']
            hidden = out.get('hidden')

            for j, i in enumerate(frame_range):
                frame_idx = i - pad
                if frame_idx not in onsets_set:
                    continue
                p = pred[j, 0]
                p_thresh = (p >= threshold).float()
                com_pred = (center_of_mass(p_thresh) if p_thresh.sum() > 0
                            else torch.zeros(2, device=device))
                com_np = com_pred.cpu().numpy()
                gt_pos = np.asarray(interpol_fnc(frame_idx))
                x_gt = float(gt_pos[1])

                x_pred_g = float(com_np[1]) + float(add_per_staff[0])
                x_gt_g = x_gt + float(add_per_staff[0])

                frame_diff = abs(float(interpol_c2o(x_pred_g)) -
                                 float(interpol_c2o(x_gt_g)))
                onset_diffs.append(frame_diff)

            t += sl

    # --- restore original conv_out weights so the next piece starts clean ---
    network.conv_out.load_state_dict(conv_out_backup)
    for p in network.parameters():
        p.requires_grad_(True)

    return onset_diffs, calib_initial_loss, calib_final_loss
