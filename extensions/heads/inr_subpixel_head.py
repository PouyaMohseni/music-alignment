"""B3 -- Sub-Pixel INR Refinement (local, not global).

CB_TA's heatmap is decoded via argmax at the U-Net's native output
resolution (page downscaled /3), introducing a coarse quantization floor on
sub-second bins (<=0.05s, <=0.1s) even when coarse localization is correct.
This is a LOCAL two-stage refinement: coarse peak from the existing heatmap
(unchanged), then a small implicit-neural-representation correction sampled
around that peak at continuous, non-tile-quantized query offsets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.hooks.film_feature_extractor import bilinear_sample, pixel_to_norm


class LocalINRRefiner(nn.Module):
    def __init__(self, feature_channels, window_px=8, hidden=128, fourier_freqs=(1, 2, 4, 8, 16)):
        super().__init__()
        self.window_px = window_px
        self.register_buffer('freqs', torch.tensor(fourier_freqs, dtype=torch.float32))
        # fourier_encode(dx,dy) -> 4*len(freqs) dims (sin/cos x2 axes)
        in_dim = feature_channels + 4 * len(fourier_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def fourier_encode(self, offsets_norm):
        """offsets_norm: (Q, 2) in [-1,1] -> (Q, 4*len(freqs))."""
        angles = offsets_norm.unsqueeze(-1) * self.freqs.view(1, 1, -1) * torch.pi   # (Q, 2, F)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-2).flatten(1)   # (Q, 4F)

    def forward(self, decoder_feature_map, coarse_peak_xy_px, query_offsets_px, score_hw):
        """decoder_feature_map: (B, C, H, W) post-FiLM feature (gradient-connected).
        coarse_peak_xy_px: (B, 2) [x, y] in ORIGINAL score-pixel space.
        query_offsets_px: (Q, 2) [dx, dy] fine sub-pixel offsets to evaluate,
            e.g. a 4x-upsampled grid within +/- window_px of the coarse peak.
        score_hw: (H_s, W_s) of the ORIGINAL score-pixel space coarse_peak_xy_px
            and query_offsets_px are expressed in.
        Returns: refined_xy_px (B, 2) = coarse_peak + soft-argmax(confidence, query_offsets)
        """
        B = decoder_feature_map.shape[0]
        Q = query_offsets_px.shape[0]

        peak_norm = pixel_to_norm(coarse_peak_xy_px, score_hw)
        local_feature = bilinear_sample(decoder_feature_map, peak_norm)   # (B, C)

        offsets_norm = query_offsets_px / self.window_px   # roughly [-1,1] within the window
        fourier = self.fourier_encode(offsets_norm)          # (Q, 4F)

        fourier_b = fourier.unsqueeze(0).expand(B, -1, -1)              # (B, Q, 4F)
        feat_b = local_feature.unsqueeze(1).expand(-1, Q, -1)            # (B, Q, C)
        combined = torch.cat([feat_b, fourier_b], dim=-1)                  # (B, Q, C+4F)
        confidence = self.mlp(combined).squeeze(-1)                        # (B, Q)

        p = F.softmax(confidence, dim=-1)
        refined_offset = (p.unsqueeze(-1) * query_offsets_px.view(1, Q, 2)).sum(dim=1)   # (B, 2)
        return coarse_peak_xy_px + refined_offset, confidence


def make_query_grid(window_px, resolution_multiplier, device):
    """(Q, 2) [dx, dy] grid spanning [-window_px, window_px] at resolution_multiplier
    points per native pixel."""
    n = int(2 * window_px * resolution_multiplier) + 1
    lin = torch.linspace(-window_px, window_px, n, device=device)
    dx, dy = torch.meshgrid(lin, lin, indexing='xy')
    return torch.stack([dx.flatten(), dy.flatten()], dim=-1)   # (Q, 2)


def heatmap_inr_loss_2d(confidence, query_offsets_px, gt_offset_px, sigma_px=5.0):
    """confidence: (B, Q) raw logits. query_offsets_px: (Q, 2). gt_offset_px:
    (B, 2) -- GT position minus coarse peak, i.e. the offset this head should
    localize. Soft cross-entropy against a 2-D isotropic Gaussian target,
    same style as the CADP M06 INR head's heatmap_inr_loss (mymodel/shared/
    losses.py) but for 2-D offsets instead of a 1-D strip position."""
    B, Q = confidence.shape
    diff = query_offsets_px.view(1, Q, 2) - gt_offset_px.view(B, 1, 2)   # (B, Q, 2)
    sq_dist = (diff ** 2).sum(-1)                                          # (B, Q)
    target = torch.exp(-0.5 * sq_dist / (sigma_px ** 2))
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    log_p = F.log_softmax(confidence, dim=-1)
    cross_entropy = -(target * log_p).sum(dim=-1)
    # Raw cross-entropy's minimum achievable value is H(target), not 0 -- since
    # target itself is a spread-out Gaussian (sigma_px=5 over a ~4225-point
    # query grid), that floor is several nats, not negligible. Subtracting it
    # turns this into KL(target || p), which is properly >= 0 and -> 0 as p
    # matches target -- without this, loss can (and did, confirmed in job
    # 64703458's log) climb for many epochs even as the refiner genuinely
    # improves, because a sharpening target (smaller gt_offset as the coarse
    # peak gets more accurate) raises H(target, p) growth faster than the
    # refiner's own convergence, even though KL is shrinking the whole time.
    target_entropy = -(target * target.clamp_min(1e-9).log()).sum(dim=-1)
    return (cross_entropy - target_entropy).mean()
