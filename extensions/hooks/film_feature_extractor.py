"""Shared hook: capture a POST-FiLM decoder feature map by its forward
output, with gradient intact -- reused by B2 (pitch aux), B3 (INR
sub-pixel), and B5 (dense contrastive), all of which need to condition an
auxiliary loss on the SAME representation the network actually uses to
localize (not a frozen/detached input -- that was the documented bug in
REDESIGN.md 9.1: the pitch head read raw precomputed embeddings with no
gradient path to anything, so the aux loss trained a dead-end side branch).

FiLM-layer numbering (from network.py) spans encoder+bottleneck+decoder as
one sequence 1..2*(n_encoder_layers+1): decoder block self.decoder[i] (i in
0..n_encoder_layers-1, 0-indexed) corresponds to combined stage
2*(n_encoder_layers+1) - (i+1). For the default n_encoder_layers=4,
film_layers=[2..8]: self.decoder[3] -> stage 6, self.decoder[2] -> stage 7,
self.decoder[1] -> stage 8, self.decoder[0] -> stage 9 (not in film_layers
list at default config, so has no FiLM -- avoid selecting it).
"""
from __future__ import annotations


def decoder_index_for_stage(n_encoder_layers: int, stage: int) -> int:
    """Map a combined FiLM-stage number (as used in CB_TA-Ext.md configs,
    e.g. 'decoder_6' -> stage=6) to self.decoder[i]'s index."""
    i_plus_1 = 2 * (n_encoder_layers + 1) - stage
    idx = i_plus_1 - 1
    if not (0 <= idx < n_encoder_layers):
        raise ValueError(f'stage {stage} does not map to a valid decoder index '
                         f'(got {idx}, valid range [0,{n_encoder_layers}))')
    return idx


class FeatureCapture:
    """Registers a forward hook on network.decoder[idx]; .feature holds the
    most recent forward's output (B, C, H, W) with autograd intact after
    each network(...) call, until the next call overwrites it."""

    def __init__(self, network, decoder_idx: int):
        self.feature = None
        self._handle = network.decoder[decoder_idx].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # ConditionalUNetBlock with down_sample=False (all decoder blocks)
        # returns a single tensor, not a tuple.
        self.feature = output

    def remove(self):
        self._handle.remove()


def bilinear_sample(feature_map, xy_norm):
    """Differentiable sampling of feature_map at continuous normalized
    coordinates. feature_map: (B, C, H, W). xy_norm: (B, 2) in [-1, 1]
    (grid_sample convention: x then y). Returns (B, C)."""
    import torch.nn.functional as F
    B = feature_map.shape[0]
    grid = xy_norm.view(B, 1, 1, 2)
    sampled = F.grid_sample(feature_map, grid, mode='bilinear', align_corners=True)  # (B,C,1,1)
    return sampled.view(B, -1)


def pixel_to_norm(xy_px, score_hw):
    """Convert a pixel coordinate in the ORIGINAL score-image pixel space to
    normalized [-1,1] coordinates, for use with bilinear_sample/grid_sample.
    Normalized coords are resolution-independent by construction (grid_sample
    interpolates relative to whatever feature map it's given), so the target
    feature map's own H,W is never needed here -- only the space xy_px is
    already expressed in.
    xy_px: (B, 2) [x, y] in score pixel space; score_hw = (H,W) of that space.
    """
    import torch
    x_px, y_px = xy_px[:, 0], xy_px[:, 1]
    H_s, W_s = score_hw
    x_norm = (x_px / max(W_s - 1, 1)) * 2 - 1
    y_norm = (y_px / max(H_s - 1, 1)) * 2 - 1
    return torch.stack([x_norm, y_norm], dim=-1)
