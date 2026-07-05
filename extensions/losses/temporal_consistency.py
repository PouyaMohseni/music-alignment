"""B4 -- Temporal Path Consistency Loss.

Dice loss is per-frame/per-pixel; nothing explicitly penalizes a decoded
position sequence that jitters or moves backward across a BPTT window. This
operates on the decoded (soft-argmax, differentiable) 2-D position sequence.
"""
import torch


def temporal_consistency_loss(pred_positions: torch.Tensor, gt_positions: torch.Tensor,
                               weight_backward: float = 2.0, weight_jerk: float = 0.1) -> torch.Tensor:
    """pred_positions, gt_positions: (T, B, 2) -- [x, y] per BPTT timestep,
    same (seq_len, batch) layout prepare_batch produces (not (B,T,2); axis 0
    is time so a monotonicity/jerk penalty over time is a diff along dim=0).

    - l1: direct position error (redundant with the dice-derived error, but
      gives gradient signal on the DECODE path specifically, since pred here
      comes from a differentiable soft-argmax, not the hard decode used at
      eval time).
    - backward_penalty: penalizes the x-coordinate moving backward in time
      (score position must not regress -- no repeats after MSMD filtering).
    - jerk_penalty: discourages frame-to-frame jitter beyond plausible tempo.
    """
    l1 = (pred_positions - gt_positions).abs().mean()

    delta = pred_positions[1:, :, 0] - pred_positions[:-1, :, 0]   # x-only monotonicity
    # last BPTT chunk of a piece can have seq_len==1 -> delta has 0 elements;
    # .mean() of an empty tensor is NaN, not 0, so this must be guarded
    # explicitly (unlike jerk_penalty below, which already was).
    backward_penalty = torch.relu(-delta).mean() if delta.numel() > 0 else delta.sum() * 0.0

    accel = delta[1:] - delta[:-1]
    jerk_penalty = accel.pow(2).mean() if accel.numel() > 0 else accel.sum() * 0.0

    return l1 + weight_backward * backward_penalty + weight_jerk * jerk_penalty
