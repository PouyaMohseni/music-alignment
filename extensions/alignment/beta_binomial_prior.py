"""Beta-binomial near-diagonal alignment prior (M1 training bootstrap).

Early in training the cross-attention alignment matrix is random, and the
forward-sum objective alone can be slow to find the monotonic ridge. The
standard fix (RAD-TTS / "One TTS Alignment To Rule Them All") is a static 2-D
beta-binomial prior: a near-diagonal matrix, wider near the centre and
narrower at the corners, multiplied into the attention scores (added in log
space) to bias the initial alignment toward the diagonal. It is annealed off
over training so the final alignment is learned, not imposed.

For a T-frame performance and an N-column score, row t (frame t) is the pmf of
a Beta-Binomial(N-1, a=scale*(t+1), b=scale*(T-t)) over column index -- whose
mode moves diagonally from column 0 (t=0) to column N-1 (t=T-1). Ported from
the NVIDIA RAD-TTS reference construction, with (frames, columns) in place of
(mel, phonemes).
"""
from __future__ import annotations
import numpy as np
import torch

try:
    from scipy.stats import betabinom as _betabinom
    _HAVE_SCIPY = True
except Exception:   # pragma: no cover - scipy is present in the cpjku venv
    _HAVE_SCIPY = False


def _betabinom_pmf_manual(k: np.ndarray, n: int, a: float, b: float) -> np.ndarray:
    """Beta-Binomial pmf via log-gamma, fallback if scipy is unavailable."""
    from math import lgamma
    out = np.empty_like(k, dtype=np.float64)
    logB_ab = lgamma(a) + lgamma(b) - lgamma(a + b)
    for i, kk in enumerate(k):
        kk = int(kk)
        log_choose = lgamma(n + 1) - lgamma(kk + 1) - lgamma(n - kk + 1)
        logB_num = lgamma(kk + a) + lgamma(n - kk + b) - lgamma(n + a + b)
        out[i] = log_choose + logB_num - logB_ab
    return np.exp(out)


def beta_binomial_prior(n_frames: int, n_cols: int, scale: float = 1.0) -> np.ndarray:
    """Returns (n_frames, n_cols) prior in PROBABILITY space; each row sums to
    ~1. `scale` sharpens (>1) or widens (<1) the diagonal band."""
    T, N = n_frames, n_cols
    cols = np.arange(N)
    rows = []
    for t in range(1, T + 1):
        a = scale * t
        b = scale * (T + 1 - t)
        if _HAVE_SCIPY:
            pmf = _betabinom(N - 1, a, b).pmf(cols)
        else:
            pmf = _betabinom_pmf_manual(cols, N - 1, a, b)
        rows.append(pmf)
    P = np.asarray(rows, dtype=np.float64)
    P = P / P.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P.astype(np.float32)


def beta_binomial_log_prior(n_frames: int, n_cols: int, scale: float = 1.0,
                            device=None, dtype=torch.float32) -> torch.Tensor:
    """Log-space prior (n_frames, n_cols) to ADD to alignment log-scores.
    Clamped away from -inf so it is safe to add before a log_softmax."""
    P = beta_binomial_prior(n_frames, n_cols, scale=scale)
    logP = np.log(np.clip(P, 1e-12, None))
    t = torch.from_numpy(logP).to(dtype=dtype)
    return t.to(device) if device is not None else t
