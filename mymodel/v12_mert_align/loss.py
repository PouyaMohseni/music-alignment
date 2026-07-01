"""
Losses for v12 MERT alignment.

InfoNCE: for each annotated onset, the correct column is the positive.
Expected-position: soft-argmax over columns, L1 vs GT position (normalised).
Combined: weighted sum of both.
"""
import torch
import torch.nn.functional as F


def infonce_loss(sim: torch.Tensor,
                 onset_frames: torch.Tensor,
                 onset_cols: torch.Tensor,
                 tau: float = 0.07) -> torch.Tensor:
    """
    sim:           (T_frames, N_cols)
    onset_frames:  (N_onsets,) int  — MERT frame indices of GT onsets
    onset_cols:    (N_onsets,) int  — GT column indices
    """
    if len(onset_frames) == 0:
        return sim.new_zeros(1).squeeze()

    # Clamp to valid range
    T, N = sim.shape
    onset_frames = onset_frames.clamp(0, T - 1)
    onset_cols   = onset_cols.clamp(0, N - 1)

    logits = sim[onset_frames] / tau           # (N_onsets, N_cols)
    loss = F.cross_entropy(logits, onset_cols)
    return loss


def expected_position_loss(sim: torch.Tensor,
                            onset_frames: torch.Tensor,
                            onset_cols: torch.Tensor,
                            tau: float = 0.1) -> torch.Tensor:
    """
    Soft-argmax expected column vs GT column, L1.
    """
    if len(onset_frames) == 0:
        return sim.new_zeros(1).squeeze()

    T, N = sim.shape
    onset_frames = onset_frames.clamp(0, T - 1)
    onset_cols   = onset_cols.clamp(0, N - 1).float()

    col_idx = torch.arange(N, device=sim.device, dtype=torch.float32)
    probs   = F.softmax(sim[onset_frames] / tau, dim=-1)   # (N_onsets, N_cols)
    expected = (probs * col_idx).sum(dim=-1)               # (N_onsets,)
    loss = F.l1_loss(expected, onset_cols) / N             # normalise by strip width
    return loss


def alignment_loss(sim, onset_frames, onset_cols,
                   w_infonce=1.0, w_expected=0.5, tau=0.07):
    l1 = infonce_loss(sim, onset_frames, onset_cols, tau=tau)
    l2 = expected_position_loss(sim, onset_frames, onset_cols, tau=tau * 1.5)
    return w_infonce * l1 + w_expected * l2, {'infonce': l1.item(), 'expected': l2.item()}
