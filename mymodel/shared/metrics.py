"""Evaluation metrics shared across model versions."""
from __future__ import annotations

import numpy as np
import torch


def tracking_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean absolute tracking error in beats (or seconds)."""
    return (pred - gt).abs().mean().item()


def percent_within_threshold(pred: torch.Tensor,
                              gt: torch.Tensor,
                              threshold: float = 0.5) -> float:
    """% of frames tracked within `threshold` beats of ground truth."""
    return ((pred - gt).abs() < threshold).float().mean().item() * 100


def recall_at_k(sim_matrix: torch.Tensor, k: int = 1) -> float:
    """Retrieval Recall@k given a (N x N) similarity matrix."""
    n = sim_matrix.size(0)
    ranks = sim_matrix.argsort(dim=-1, descending=True)
    correct = (ranks[:, :k] == torch.arange(n).unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def dtw_backtrack(sim: np.ndarray, band_radius_frac: float | None = None) -> np.ndarray:
    """Hard DTW backtrack that MAXIMIZES sim. Returns (T, 2) int path [(t, n), ...].

    sim: (T, N) similarity, higher is better.
    """
    T, N = sim.shape
    cost = -sim.astype(np.float64)
    INF = np.inf

    if band_radius_frac is not None:
        r = max(1, int(max(T, N) * band_radius_frac))
        i = np.arange(T)[:, None].astype(np.float64)
        j = np.arange(N)[None, :].astype(np.float64)
        proj_j = i * ((N - 1) / max(T - 1, 1))
        cost = np.where(np.abs(j - proj_j) <= r, cost, INF)

    D = np.full((T + 1, N + 1), INF)
    D[0, 0] = 0.0
    bt = np.zeros((T, N), dtype=np.int8)  # 0=diag, 1=up, 2=left
    for t in range(1, T + 1):
        for n in range(1, N + 1):
            diag = D[t - 1, n - 1]
            up   = D[t - 1, n]
            left = D[t, n - 1]
            best = diag
            choice = 0
            if up < best:
                best, choice = up, 1
            if left < best:
                best, choice = left, 2
            D[t, n] = cost[t - 1, n - 1] + best
            bt[t - 1, n - 1] = choice

    path = []
    t, n = T - 1, N - 1
    while t > 0 or n > 0:
        path.append((t, n))
        c = bt[t, n]
        if c == 0 and t > 0 and n > 0:
            t, n = t - 1, n - 1
        elif c == 1 and t > 0:
            t -= 1
        elif c == 2 and n > 0:
            n -= 1
        elif t > 0:
            t -= 1
        else:
            n -= 1
    path.append((0, 0))
    return np.array(path[::-1], dtype=np.int64)


def alignment_metrics(pred_strip_x_at_onset: np.ndarray,
                       gt_strip_x: np.ndarray,
                       pixels_per_sec: float,
                       thresholds_sec: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)) -> dict:
    """Per-notehead tracking metrics.

    pred_strip_x_at_onset : (K,) predicted strip-x at each GT onset time
    gt_strip_x            : (K,) ground-truth strip-x of the notehead
    pixels_per_sec        : scalar, used to convert px error to sec error

    Returns mean_abs_err_px, mean_abs_err_sec, pct_within_<thr> for each thr.
    """
    err_px = np.abs(pred_strip_x_at_onset - gt_strip_x).astype(np.float64)
    err_sec = err_px / float(pixels_per_sec)
    out = {
        "n": int(len(gt_strip_x)),
        "mean_abs_err_px":  float(err_px.mean()) if len(err_px) else float("nan"),
        "median_abs_err_px": float(np.median(err_px)) if len(err_px) else float("nan"),
        "mean_abs_err_sec": float(err_sec.mean()) if len(err_sec) else float("nan"),
        "median_abs_err_sec": float(np.median(err_sec)) if len(err_sec) else float("nan"),
    }
    for thr in thresholds_sec:
        out[f"pct_within_{thr}s"] = float((err_sec < thr).mean()) * 100.0 if len(err_sec) else float("nan")
    return out
