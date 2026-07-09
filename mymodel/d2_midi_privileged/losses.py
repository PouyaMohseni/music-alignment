"""D2 losses -- both privileged-information terms, MIDI-train-only.

- soft_multi_target_ce_loss: D1's dense_ce_loss, generalized to a mixture of
  Gaussian targets (true column + MIDI-discovered repeat alternates) instead
  of a single-Gaussian target. Degenerates exactly to dense_ce_loss when a
  frame's onset has no repeat alternates.
- midi_distill_loss: symmetric InfoNCE between the audio tower's per-frame
  embedding and the MIDI encoder's per-frame embedding (same recipe as C4's
  tempo-contrastive InfoNCE, applied cross-modally). Training-only signal;
  MidiEncoder is discarded at inference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_multi_target_ce_loss(S: torch.Tensor, gt_cols: torch.Tensor,
                              repeat_alt_cols: list[list[int]] | None = None,
                              sigma_cols: float = 3.0, alt_weight: float = 0.3,
                              valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """S: (T, W_col). gt_cols: (T,) long, true GT column per frame.
    repeat_alt_cols: length-T list, repeat_alt_cols[t] = list of alternate
    columns for frame t's onset (empty list if none / not an onset frame).
    Mixture target: weight 1.0 Gaussian on gt_cols[t], weight `alt_weight`
    Gaussian on each alternate -- so an alternate is "not punished as harshly",
    not "equally correct" (the true column still gets more mass whenever the
    two Gaussians don't fully overlap)."""
    T, W = S.shape
    cols = torch.arange(W, device=S.device, dtype=torch.float32).unsqueeze(0)   # (1, W)

    def gauss(center_col: float) -> torch.Tensor:
        return torch.exp(-0.5 * ((cols - center_col) / sigma_cols) ** 2)        # (1, W)

    target = torch.zeros(T, W, device=S.device)
    gt_f = gt_cols.to(torch.float32)
    for t in range(T):
        target[t] = gauss(float(gt_f[t])).squeeze(0)
        alts = repeat_alt_cols[t] if repeat_alt_cols is not None else []
        for c in alts:
            target[t] = target[t] + alt_weight * gauss(float(c)).squeeze(0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    log_p = F.log_softmax(S, dim=-1)
    ce = -(target * log_p).sum(dim=-1)
    if valid_mask is not None:
        ce = ce[valid_mask]
        if ce.numel() == 0:
            return S.new_zeros(())
    return ce.mean()


def midi_distill_loss(A: torch.Tensor, M: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """A: (T, d) audio tower embeddings. M: (T, d) MIDI encoder embeddings,
    same frames. Symmetric InfoNCE, frame t's audio embedding should match
    frame t's MIDI embedding against all other frames' MIDI embeddings in the
    piece (and vice versa)."""
    T = A.shape[0]
    if T < 2:
        return A.new_zeros(())
    logits = (A @ M.t()) / temperature   # (T, T)
    labels = torch.arange(T, device=A.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_m = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_m) / 2
