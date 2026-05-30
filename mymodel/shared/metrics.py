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


def alignment_metrics(
    pred_strip_x_at_onset: np.ndarray,
    gt_strip_x: np.ndarray,
    pixels_per_sec: float,
    thresholds_sec: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    beat_times_sec: list[float] | None = None,
    bar_times_sec: list[float] | None = None,
    gt_onset_sec: np.ndarray | None = None,
    thresholds_beats: tuple[float, ...] = (0.5, 1.0),
) -> dict:
    """Per-notehead tracking metrics.

    Required:
        pred_strip_x_at_onset : (K,) predicted strip-x at each GT onset
        gt_strip_x            : (K,) ground-truth strip-x
        pixels_per_sec        : strip width / audio duration

    Optional (enable beat/bar metrics):
        beat_times_sec  : list of beat boundary times in seconds
        bar_times_sec   : list of bar boundary times in seconds
        gt_onset_sec    : (K,) GT onset times in seconds (needed for beat/bar metrics)
        thresholds_beats: thresholds for pct_within_X_beats

    Always returns:
        n, mean/median_abs_err_px, mean/median_abs_err_sec,
        pct_within_{0.1,0.25,0.5,1.0}s

    Also returns when beat/bar arrays provided:
        mean/median_abs_err_beats, pct_within_{0.5,1.0}_beats,
        mean/median_abs_err_bars
    """
    err_px = np.abs(pred_strip_x_at_onset - gt_strip_x).astype(np.float64)
    err_sec = err_px / float(pixels_per_sec)

    out: dict = {
        "n": int(len(gt_strip_x)),
        "mean_abs_err_px":    float(err_px.mean())    if len(err_px) else float("nan"),
        "median_abs_err_px":  float(np.median(err_px)) if len(err_px) else float("nan"),
        "mean_abs_err_sec":   float(err_sec.mean())   if len(err_sec) else float("nan"),
        "median_abs_err_sec": float(np.median(err_sec)) if len(err_sec) else float("nan"),
    }
    for thr in thresholds_sec:
        out[f"pct_within_{thr}s"] = float((err_sec < thr).mean()) * 100.0 if len(err_sec) else float("nan")

    # ---- beat-level metrics (Henkel 2019) ----
    if beat_times_sec is not None and gt_onset_sec is not None and len(beat_times_sec) > 1:
        beats = np.asarray(beat_times_sec, dtype=np.float64)

        def sec_to_beat(t_sec: np.ndarray) -> np.ndarray:
            """Map absolute times in seconds to fractional beat index."""
            idx = np.searchsorted(beats, t_sec, side="right") - 1
            idx = np.clip(idx, 0, len(beats) - 2)
            seg_start = beats[idx]
            seg_len   = beats[idx + 1] - beats[idx]
            frac = np.where(seg_len > 0, (t_sec - seg_start) / seg_len, 0.0)
            return idx.astype(np.float64) + frac

        # GT and predicted positions both converted to beat-time
        gt_beat   = sec_to_beat(gt_onset_sec.astype(np.float64))
        pred_sec  = pred_strip_x_at_onset.astype(np.float64) / float(pixels_per_sec)
        pred_beat = sec_to_beat(pred_sec)
        err_beats = np.abs(pred_beat - gt_beat)

        out["mean_abs_err_beats"]   = float(err_beats.mean())
        out["median_abs_err_beats"] = float(np.median(err_beats))
        for thr in thresholds_beats:
            out[f"pct_within_{thr}_beats"] = float((err_beats < thr).mean()) * 100.0

    # ---- bar-level metrics ----
    if bar_times_sec is not None and gt_onset_sec is not None and len(bar_times_sec) > 1:
        bars = np.asarray(bar_times_sec, dtype=np.float64)

        def sec_to_bar(t_sec: np.ndarray) -> np.ndarray:
            idx = np.searchsorted(bars, t_sec, side="right") - 1
            idx = np.clip(idx, 0, len(bars) - 2)
            seg_start = bars[idx]
            seg_len   = bars[idx + 1] - bars[idx]
            frac = np.where(seg_len > 0, (t_sec - seg_start) / seg_len, 0.0)
            return idx.astype(np.float64) + frac

        gt_bar   = sec_to_bar(gt_onset_sec.astype(np.float64))
        pred_sec = pred_strip_x_at_onset.astype(np.float64) / float(pixels_per_sec)
        pred_bar = sec_to_bar(pred_sec)
        err_bars = np.abs(pred_bar - gt_bar)

        out["mean_abs_err_bars"]   = float(err_bars.mean())
        out["median_abs_err_bars"] = float(np.median(err_bars))

    return out


def retrieval_metrics(sim: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Recall@k for audio→score retrieval from a (T_audio, N_tiles) sim matrix.

    Each audio frame should retrieve the correct tile as its nearest neighbour.
    Correct tile = the one whose centre is closest in strip-x to the GT position.

    Args:
        sim: (T, N) similarity matrix (higher = more similar)
        ks:  recall thresholds to evaluate

    Returns dict with recall_at_1, recall_at_5, recall_at_10.
    """
    T, N = sim.shape
    # For each audio frame t, the "correct" tile is the argmax along its GT diagonal.
    # Here we compute a simpler retrieval metric: for each row, whether the
    # top-k retrieved tiles include the diagonal tile (the tile at position
    # round(t * (N-1) / (T-1))).
    gt_tile = np.round(np.arange(T) * (N - 1) / max(T - 1, 1)).astype(np.int64)
    ranks = np.argsort(-sim, axis=1)                                    # (T, N) descending
    out = {}
    for k in ks:
        top_k = ranks[:, :k]                                            # (T, k)
        correct = (top_k == gt_tile[:, None]).any(axis=1)               # (T,)
        out[f"recall_at_{k}"] = float(correct.mean()) * 100.0
    return out
