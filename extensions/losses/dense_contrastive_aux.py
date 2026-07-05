"""B5 -- Dense Contrastive Deep Supervision.

Adapts DenseAV's (Hamilton et al., CVPR 2024) bidirectional token-alignment
idea to an architecture with no similarity matrix: an auxiliary loss at an
intermediate FiLM-modulated feature map -- the feature vector at the
ground-truth location should be more similar to the audio embedding (the
LSTM/rnn output, i.e. the FiLM-conditioning vector) than feature vectors at
other sampled locations.
"""
import torch
import torch.nn.functional as F

from extensions.hooks.film_feature_extractor import bilinear_sample, pixel_to_norm


def sample_random_coords(H, W, gt_xy_px, exclude_radius_px, n, device):
    """gt_xy_px: (B, 2) [x, y] in feature-map pixel space. Returns (B, n, 2)
    random [x, y] pixel coords, resampled away from gt if within exclude_radius_px
    (best-effort: a few retries, not a hard guarantee for tiny feature maps)."""
    B = gt_xy_px.shape[0]
    coords = torch.stack([
        torch.randint(0, W, (B, n), device=device).float(),
        torch.randint(0, H, (B, n), device=device).float(),
    ], dim=-1)   # (B, n, 2)
    for _ in range(3):
        dist = (coords - gt_xy_px.unsqueeze(1)).norm(dim=-1)   # (B, n)
        too_close = dist < exclude_radius_px
        if not too_close.any():
            break
        resample = torch.stack([
            torch.randint(0, W, (B, n), device=device).float(),
            torch.randint(0, H, (B, n), device=device).float(),
        ], dim=-1)
        coords = torch.where(too_close.unsqueeze(-1), resample, coords)
    return coords


def dense_contrastive_aux_loss(film_feature_map, rnn_out, gt_xy_score_px, score_hw,
                                audio_proj: torch.nn.Module,
                                num_negatives=32, exclude_radius_px=30, temperature=0.07):
    """film_feature_map: (B, C, H, W) post-FiLM decoder feature (gradient-connected).
    rnn_out: (B, rnn_size) -- the FiLM-conditioning vector (aligned representation).
    gt_xy_score_px: (B, 2) [x, y] in ORIGINAL score-pixel space (will be mapped
    to this feature map's own H,W via pixel_to_norm/grid_sample).
    audio_proj: nn.Linear(rnn_size, C) -- trainable, projects rnn_out to the
    feature map's channel dim so cosine similarity is well-defined.
    """
    B, C, H, W = film_feature_map.shape
    device = film_feature_map.device

    a = F.normalize(audio_proj(rnn_out), dim=-1)                     # (B, C)

    gt_norm = pixel_to_norm(gt_xy_score_px, score_hw)
    positive_feature = F.normalize(bilinear_sample(film_feature_map, gt_norm), dim=-1)   # (B, C)
    positive_sim = (a * positive_feature).sum(-1)                     # (B,)

    # negatives sampled in FEATURE-MAP pixel space (H,W), excluded near the
    # GT location (mapped into that same feature-space), converted to [-1,1]
    gt_xy_feat = torch.stack([
        gt_xy_score_px[:, 0] / max(score_hw[1] - 1, 1) * (W - 1),
        gt_xy_score_px[:, 1] / max(score_hw[0] - 1, 1) * (H - 1),
    ], dim=-1)
    neg_coords_feat = sample_random_coords(H, W, gt_xy_feat, exclude_radius_px, num_negatives, device)

    neg_norm = torch.stack([
        neg_coords_feat[..., 0] / max(W - 1, 1) * 2 - 1,
        neg_coords_feat[..., 1] / max(H - 1, 1) * 2 - 1,
    ], dim=-1)   # (B, n, 2)

    neg_features = []
    for k in range(num_negatives):
        nf = F.normalize(bilinear_sample(film_feature_map, neg_norm[:, k]), dim=-1)   # (B, C)
        neg_features.append(nf)
    neg_features = torch.stack(neg_features, dim=1)   # (B, n, C)
    negative_sims = torch.einsum('bc,bnc->bn', a, neg_features)

    logits = torch.cat([positive_sim.unsqueeze(1), negative_sims], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)
