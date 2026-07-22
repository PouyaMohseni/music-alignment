"""Monotonic Viterbi decode + position readout (M1 inference).

The training-time forward-sum (forward_sum.py) sums over all monotonic paths;
at inference we want the single most-likely one. This is the same stay-or-
advance-by-one lattice with max-plus instead of log-sum-exp, plus a backtrace.
Because the decoded path is forced monotonic and continuous over the WHOLE
performance, a repeated passage is resolved by global context -- the path has
already advanced past it, so the second occurrence maps to the later column,
not the earlier identical-looking one. This is the structural property a
per-frame argmax decode (Henkel / CB_TA) cannot have.
"""
from __future__ import annotations
import torch

_NEG = -1e9


def viterbi_path(log_emit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """log_emit: (T, N). Returns (path, score):
      path:  (T,) long, the most-likely column per frame, non-decreasing,
             path[0]==0, path[T-1]==N-1, consecutive diffs in {0, 1}.
      score: scalar, the summed log-emission along that best path.
    Requires T >= N.
    """
    T, N = log_emit.shape
    device, dtype = log_emit.device, log_emit.dtype

    delta = torch.full((N,), _NEG, device=device, dtype=dtype)
    delta[0] = log_emit[0, 0]
    # back[t, n] in {0,1}: 0 = stayed (came from column n), 1 = advanced (from n-1)
    back = torch.zeros((T, N), dtype=torch.long, device=device)

    for t in range(1, T):
        adv = torch.cat([delta.new_full((1,), _NEG), delta[:-1]], dim=0)   # delta_prev[n-1]
        stay = delta                                                        # delta_prev[n]
        take_adv = adv > stay
        best_prev = torch.where(take_adv, adv, stay)
        back[t] = take_adv.long()
        delta = log_emit[t] + best_prev

    # backtrace from (T-1, N-1)
    path = torch.zeros(T, dtype=torch.long, device=device)
    n = N - 1
    for t in range(T - 1, 0, -1):
        path[t] = n
        if back[t, n] == 1:      # advanced into n -> came from n-1
            n = n - 1
    path[0] = n                  # should be 0
    return path, delta[N - 1]


def expected_position(scores_or_probs: torch.Tensor, col_x: torch.Tensor,
                      apply_softmax: bool = True) -> torch.Tensor:
    """Soft (differentiable) position readout: per-frame expected x-coordinate
    under the column posterior. scores_or_probs: (T, N). col_x: (N,) the
    x-pixel coordinate of each column. Returns (T,) expected x per frame.
    Ignores monotonicity -- this is the "raw posterior" readout; use
    viterbi_path + path_to_position for the monotone-decoded readout.
    """
    P = torch.softmax(scores_or_probs, dim=1) if apply_softmax else scores_or_probs
    return P @ col_x


def path_to_position(path: torch.Tensor, col_x: torch.Tensor) -> torch.Tensor:
    """Hard position readout from a decoded column path. path: (T,) long column
    indices. col_x: (N,) column x-coordinates. Returns (T,) x per frame."""
    return col_x[path]
