"""Forward-sum monotonic-alignment loss (the core M1 training objective).

Given a per-frame log-emission matrix log_emit[t, n] = log P(audio frame t is
emitted by score column n), the forward-sum objective is the log-likelihood
of ALL monotonic alignment paths, summed (not maxed), computed by the
classic left-to-right forward recursion. Minimising -logZ pushes probability
mass onto monotonic paths without ever committing to a single one during
training -- the model learns an alignment posterior, and monotonicity is a
hard structural property of that posterior rather than a soft penalty.

Path model (the "stay-or-advance-by-one" lattice, == CTC-without-blank ==
the DTW step set restricted to a per-frame time step):
  - time advances by exactly 1 each step (one column assignment per frame),
  - column advances by 0 (stay) or +1 (advance) each step,
  - the path starts at column 0 (frame 0 <-> first onset column) and ends at
    column N-1 (last frame <-> last onset column).
This is monotonic AND surjective: every score column is visited, in order.
It is the correct model when the "columns" axis is the sequence of ground-
truth onset anchors (each must be hit, in order) -- which is exactly how the
M1 target axis is defined. (For a repeated section, the onset anchors live on
the UNROLLED score axis, on which the path IS monotone; see repeat_unroll.py.)

References: Badlani et al., "One TTS Alignment To Rule Them All" (2021);
Mensch & Blondel, "Differentiable Dynamic Programming" (2018). Numerics
(finite negative sentinel instead of true -inf, to keep logaddexp gradients
finite) mirror mymodel/d1_align_matrix/losses.py's `_INF` convention.
"""
from __future__ import annotations
import torch

# Finite stand-in for log(0). Large enough that exp(_NEG) underflows to 0 in
# float32 (so unreachable cells contribute nothing to any logsumexp), but
# finite so torch.logaddexp(_NEG, _NEG) yields a finite value with finite
# gradient -- true -inf would produce NaN gradients on all-unreachable rows.
_NEG = -1e9


def forward_sum_logZ(log_emit: torch.Tensor) -> torch.Tensor:
    """log_emit: (T, N) log-emission scores (log P(frame t | column n), or any
    per-frame log-scores -- the DP is identical). Returns the scalar log-sum-
    exp over all stay-or-advance-by-one monotonic paths from (t=0, n=0) to
    (t=T-1, n=N-1) of the summed log-emissions along the path.

    Requires T >= N (need at least N frames to advance through N columns). If
    T < N the end cell is unreachable and logZ == _NEG (caller should guard).
    """
    T, N = log_emit.shape

    # alpha[n] = logsumexp over monotonic paths of frames 0..t ending at column n.
    # Built by concatenation (not in-place assignment) so autograd/gradcheck
    # flow cleanly through log_emit[0, 0]; must start at column 0.
    alpha = torch.cat([log_emit[0, :1], log_emit.new_full((N - 1,), _NEG)], dim=0)

    for t in range(1, T):
        # advance term: alpha_prev[n-1] (came from column n-1); n=0 has no
        # predecessor to advance from -> _NEG. Shift alpha right by one column.
        adv = torch.cat([alpha.new_full((1,), _NEG), alpha[:-1]], dim=0)
        stay = alpha                                # alpha_prev[n]
        alpha = log_emit[t] + torch.logaddexp(stay, adv)

    return alpha[N - 1]


def forward_sum_loss(scores: torch.Tensor, apply_log_softmax: bool = True,
                     normalize: bool = True) -> torch.Tensor:
    """Training-facing wrapper. scores: (T, N) raw alignment scores (e.g.
    audio-frame x score-column cross-attention logits). With apply_log_softmax
    (default), each frame's scores are turned into a proper distribution over
    columns before the forward-sum, so -logZ is a genuine negative
    log-likelihood. normalize divides by T -> mean NLL per frame (comparable
    across pieces of different length), matching d1_align_matrix's
    normalize-by-path-length convention.
    """
    T, N = scores.shape
    if T < N:
        raise ValueError(f"forward_sum_loss needs T>=N (frames>=columns); got T={T}, N={N}")
    log_emit = torch.log_softmax(scores, dim=1) if apply_log_softmax else scores
    logZ = forward_sum_logZ(log_emit)
    nll = -logZ
    return nll / T if normalize else nll


def forward_sum_loss_batched(scores: torch.Tensor, frame_lens: torch.Tensor,
                             col_lens: torch.Tensor, apply_log_softmax: bool = True,
                             normalize: bool = True) -> torch.Tensor:
    """Batched mean over a (B, T_max, N_max) padded batch with per-item true
    lengths. Loops over the batch calling the single-item core on each valid
    slice -- Phase-0 clarity over throughput; a fully-vectorised batched DP is
    a Phase-1 optimisation. Returns the mean loss over the batch.
    """
    B = scores.shape[0]
    losses = []
    for b in range(B):
        T_b, N_b = int(frame_lens[b]), int(col_lens[b])
        losses.append(forward_sum_loss(scores[b, :T_b, :N_b],
                                       apply_log_softmax=apply_log_softmax, normalize=normalize))
    return torch.stack(losses).mean()
