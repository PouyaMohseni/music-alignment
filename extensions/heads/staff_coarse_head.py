"""A3 -- coarse-to-fine: classify the STAFF, then localise within it.

WHY, FROM OUR OWN EVIDENCE
--------------------------
Two independent things say vertical localisation is a distinct sub-problem that
none of our models treats as one:

  1. `cyolo_sb` beats `cyolo` by +4.9 on real audio, and the only difference is
     that it also predicts bar and system boxes -- coarse structure -- alongside
     the note.
  2. The first P1 run scored 10.6 on room for exactly one reason: its
     prediction was accurate in x but always landed on the MIDDLE staff, so the
     metric unrolled every x against the wrong staff offset. The column was
     right; the staff was wrong. A model whose x is good and whose staff is bad
     scores near zero, which means staff assignment carries an enormous share of
     the metric and we have never supervised it directly.

Both point the same way: give the network an explicit, low-cardinality decision
("which of the ~6 staff rows am I on?") instead of asking a dense heatmap to get
it right implicitly.

MECHANISM
---------
An auxiliary head on the decoder features predicts a distribution over staff
ROWS (a coarse quantisation of y, not a per-pixel map). At inference the fine
position's y-logits are biased by lambda * log p(staff), which sharpens the
vertical decision without overriding it -- the same gating idea that
`cyolo_sb`'s coarse heads apply to its fine head.

The target needs NO new data: it is derived by pooling the existing GT mask over
x and quantising y into `n_staff_bins` bands, so this is a supervision change
rather than a data change and can run on every existing experiment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaffCoarseHead(nn.Module):
    """Decoder features -> distribution over coarse y bands.

    in_ch: channels of the decoder stage this is attached to.
    """

    def __init__(self, in_ch: int, n_bins: int = 16, hidden: int = 64):
        super().__init__()
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.ELU(False),
        )
        # Per-BAND scoring, not a global Linear. See forward().
        self.out = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (N, C, H, W) -> (N, n_bins) logits.

        THE Y AXIS MUST SURVIVE. An earlier version did

            h = h.mean(dim=3)   # pool over x -- correct, staff is a y question
            h = h.mean(dim=2)   # pool over y -- destroys the answer

        Convolutions are translation-equivariant and there is no positional
        encoding, so averaging over y leaves nothing that depends on WHERE the
        evidence was. Measured on the real feature shape: the same blob at rows
        3-8 and at rows 40-45 gave the same argmax band, max |logit diff| 6e-8.
        The head was position-blind by construction while its own docstring
        claimed to fix exactly that failure.

        Now: pool over x only, then pool the y axis down to n_bins and score
        each band from its own features, so band k's logit is a function of the
        features at band k.
        """
        h = self.net(feat)                     # (N, hidden, H, W)
        h = h.mean(dim=3)                      # pool x only -> (N, hidden, H)
        h = F.adaptive_avg_pool1d(h, self.n_bins)   # -> (N, hidden, n_bins)
        return self.out(h)[:, 0]               # -> (N, n_bins)


def staff_target(y: torch.Tensor, n_bins: int, eps: float = 1e-8):
    """(N,1,H,W) GT mask -> ((N, n_bins) soft target, (N,) valid).

    Pools the mask over x and average-pools y into n_bins bands.  Soft rather
    than argmax so a blob straddling a band boundary is not forced to lie.
    """
    n, _, h, w = y.shape
    prof = y[:, 0].sum(dim=2)                                  # (N, H) over x
    # adaptive pooling handles H not divisible by n_bins
    binned = F.adaptive_avg_pool1d(prof.unsqueeze(1), n_bins).squeeze(1)   # (N, n_bins)
    total = binned.sum(dim=1, keepdim=True)
    valid = (total.squeeze(1) > eps)
    q = torch.where(total > eps, binned / total.clamp_min(eps), torch.zeros_like(binned))
    return q, valid


def staff_loss(logits: torch.Tensor, y: torch.Tensor, n_bins: int):
    """Soft cross-entropy over coarse y bands. Returns (loss, n_valid)."""
    q, valid = staff_target(y, n_bins)
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0
    logp = F.log_softmax(logits, dim=1)
    ce = -(q[valid] * logp[valid]).sum(dim=1)
    return ce.mean(), int(valid.sum().item())


def bias_fine_logits(fine_logits: torch.Tensor, staff_logits: torch.Tensor,
                     lam: float = 0.5) -> torch.Tensor:
    """Bias per-pixel logits by lambda * log p(coarse band) at inference.

    fine_logits: (N,1,H,W); staff_logits: (N, n_bins).
    Upsampled nearest so each row inherits its band's log-probability. lam<1
    keeps this a prior, not an override -- if the coarse head is confidently
    wrong we still want the fine evidence to be able to win.
    """
    n, _, h, w = fine_logits.shape
    logp = F.log_softmax(staff_logits, dim=1)                       # (N, bins)
    up = F.interpolate(logp.unsqueeze(1), size=h, mode='nearest')   # (N,1,H)
    return fine_logits + lam * up.unsqueeze(-1)
