"""Shared position-decode utilities for B2-B5's auxiliary losses.

GT position is derived from the GT heatmap's (y_batch) center-of-mass rather
than the dataset's stored `true_positions` field, because `true_positions` is
NOT updated by ScoreAudioDataset's random yshift/xshift augmentation (only
the heatmap/score images are `np.roll`ed) -- using it directly would silently
feed stale, pre-shift coordinates into any auxiliary loss whenever
--augment is on (which both A0 and every B-extension use). Center-of-mass on
the already-shifted heatmap sidesteps this and needs no changes to the base
dataset pipeline.
"""
from __future__ import annotations
import torch


def center_of_mass_xy(heatmap: torch.Tensor) -> torch.Tensor:
    """heatmap: (..., H, W) -> (..., 2) as [x, y] (col, row) in pixel space.
    Zero-mass frames (no annotation in this window) return the image center."""
    *lead, H, W = heatmap.shape
    flat = heatmap.reshape(-1, H, W)
    mass = flat.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    ys = torch.arange(H, device=heatmap.device, dtype=heatmap.dtype).view(1, H, 1)
    xs = torch.arange(W, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, W)
    y = (flat * ys).sum(dim=(-2, -1), keepdim=True) / mass
    x = (flat * xs).sum(dim=(-2, -1), keepdim=True) / mass
    xy = torch.stack([x.view(-1), y.view(-1)], dim=-1)   # (N, 2)
    return xy.view(*lead, 2)


def soft_argmax_xy(pred: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """pred: (..., H, W) raw sigmoid segmentation output -> differentiable
    (..., 2) [x, y] position via softmax-weighted centroid. Used for B4's
    temporal-consistency loss, which needs the DECODED path to be
    differentiable (soft-argmax), not hard argmax."""
    *lead, H, W = pred.shape
    flat = pred.reshape(-1, H * W)
    weights = torch.softmax(flat / temperature, dim=-1).view(-1, H, W)
    return center_of_mass_xy(weights).view(*lead, 2)
