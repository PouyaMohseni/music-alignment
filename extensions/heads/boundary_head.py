"""A4 -- boundary-oriented position prediction (BAM-DETR adapted).

WHY, AND WHERE IT COMES FROM
----------------------------
BAM-DETR (ECCV 2024, arXiv:2312.00083) diagnoses a problem in query-based
localisation that reads like a description of ours:

    "DETR-like approaches predict the CENTER and length of a target moment, but
     they suffer from the issue of CENTER MISALIGNMENT raised by the inherent
     ambiguity of moment centers, leading to inaccurate predictions."

Their fix is to stop predicting the center. The model instead predicts ANY
ANCHOR POINT INSIDE the interval -- a much better-posed target, because it is
satisfied by a set of positions rather than one -- and then estimates the two
BOUNDARIES as offsets from that anchor. They report their largest gains at
TIGHT tolerances, which is exactly where we are weakest: our best model scores
56.6 at 0.5 s but only 32.9 / 34.6 at 0.05 / 0.1 s.

Every localisation output we have tried predicts a center, one way or another:
the soft-Dice heatmap's argmax, the bucketed softmax's argmax, CYOLO's box
center. This is the first formulation that does not.

Three independent measurements of ours point at the output layer as the binding
constraint, which is why this is worth building rather than more conditioning
capacity:
  * detection output 67.1 on room vs our heatmap 56.6, same IR bank;
  * MM-Loc 58.5 vs CUNet 22.4, same paper/data/audio tower;
  * real-IR augmentation worth +25.2 to a detector but +11.0 to our heatmap.

THE ADAPTATION
--------------
In temporal grounding the "moment" is an interval in time. Here the interval is
in SPACE: at any audio frame the sounding musical content occupies an x-span on
the score, and the GT mask already has that extent. So per frame we predict

    anchor_logits  (W,)   p(column c lies INSIDE the sounding span)
    d_left         (W,)   from column c, normalised distance to the left edge
    d_right        (W,)   from column c, normalised distance to the right edge

and decode  x_hat = midpoint( anchor - d_left[anchor], anchor + d_right[anchor] ).

That midpoint is still a single number, but it is DERIVED from two well-posed
edge estimates instead of being regressed directly as an ill-posed center --
which is precisely BAM-DETR's argument. Boundaries are also where the visual
evidence is sharpest: a notehead has edges, its middle looks like its
neighbours' middles.

Pairs with extensions/heads/staff_coarse_head.py for the y dimension: that
handles WHICH STAFF, this handles WHERE ALONG IT.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHead(nn.Module):
    """Decoder features -> (anchor_logits, d_left, d_right), each (N, W).

    Distances are predicted in NORMALISED units (fraction of width) through a
    softplus, so they are non-negative by construction and scale-free across
    strips of different widths.
    """

    def __init__(self, in_ch: int, hidden: int = 96):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, hidden), nn.ELU(False),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, hidden), nn.ELU(False),
        )
        # separate 1-D branches over columns: the anchor is a classification and
        # the offsets are regressions, and sharing a final layer between them
        # made the offsets fight the logit scale in early experiments
        self.anchor = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)
        self.offsets = nn.Conv1d(hidden, 2, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor):
        h = self.trunk(feat)                  # (N, C, H, W)
        h = h.mean(dim=2)                     # collapse y -> (N, C, W)
        a = self.anchor(h)[:, 0]              # (N, W)
        o = F.softplus(self.offsets(h))       # (N, 2, W), >= 0
        return a, o[:, 0], o[:, 1]


def span_targets(y: torch.Tensor, rel_thresh: float = 0.5, eps: float = 1e-8):
    """(N,1,H,W) GT mask -> anchor target, normalised d_left/d_right, masks.

    The span is the set of columns whose x-profile exceeds `rel_thresh` of its
    own maximum -- a half-maximum width, which is stable against the blob's
    absolute amplitude.

    Returns (q_anchor, dl, dr, inside, valid):
      q_anchor (N,W) normalised distribution over INSIDE columns (uniform on the
                    span, not a peak: any interior anchor is equally correct,
                    which is the whole point of the reformulation)
      dl, dr   (N,W) normalised distance from each column to each edge
      inside   (N,W) bool, where the offset regression is supervised
      valid    (N,)  frames with a non-empty span
    """
    n, _, h, w = y.shape
    prof = y[:, 0].sum(dim=1)                                  # (N, W) over y
    mx = prof.max(dim=1, keepdim=True).values
    valid = (mx.squeeze(1) > eps)
    inside = prof >= (rel_thresh * mx.clamp_min(eps))
    inside = inside & valid[:, None]

    cols = torch.arange(w, device=y.device).float()[None, :].expand(n, w)
    big = torch.full_like(cols, float(w))
    xl = torch.where(inside, cols, big).min(dim=1).values          # (N,)
    xr = torch.where(inside, cols, torch.full_like(cols, -1.0)).max(dim=1).values

    dl = (cols - xl[:, None]).clamp_min(0.0) / w
    dr = (xr[:, None] - cols).clamp_min(0.0) / w

    cnt = inside.sum(dim=1, keepdim=True).clamp_min(1)
    q = inside.float() / cnt
    q = torch.where(valid[:, None], q, torch.zeros_like(q))
    return q, dl, dr, inside, valid


def boundary_loss(anchor_logits: torch.Tensor, dl_pred: torch.Tensor,
                  dr_pred: torch.Tensor, y: torch.Tensor,
                  w_anchor: float = 1.0, w_off: float = 5.0, w_iou: float = 2.0):
    """Anchor CE + offset L1 (inside only) + 1-D IoU on the decoded interval.

    The IoU term matters: minimising the two offsets independently can leave a
    systematically too-wide or too-narrow interval whose midpoint is still
    biased, and IoU is the term that couples them.
    """
    q, dl_t, dr_t, inside, valid = span_targets(y)
    if valid.sum() == 0:
        z = anchor_logits.sum() * 0.0
        return z, {'anchor': 0.0, 'off': 0.0, 'iou': 0.0, 'n': 0}

    logp = F.log_softmax(anchor_logits, dim=1)
    l_anchor = -(q[valid] * logp[valid]).sum(dim=1).mean()

    m = inside[valid].float()
    denom = m.sum().clamp_min(1.0)
    l_off = (((dl_pred[valid] - dl_t[valid]).abs()
              + (dr_pred[valid] - dr_t[valid]).abs()) * m).sum() / denom

    # 1-D IoU between predicted and true intervals, at the true anchor columns
    cols = torch.arange(y.shape[-1], device=y.device).float()[None, :]
    p_l = cols - dl_pred * y.shape[-1]
    p_r = cols + dr_pred * y.shape[-1]
    t_l = cols - dl_t * y.shape[-1]
    t_r = cols + dr_t * y.shape[-1]
    inter = (torch.minimum(p_r, t_r) - torch.maximum(p_l, t_l)).clamp_min(0)
    union = (torch.maximum(p_r, t_r) - torch.minimum(p_l, t_l)).clamp_min(1e-6)
    iou = (inter / union)[valid]
    l_iou = (1.0 - iou).mul(m).sum() / denom

    total = w_anchor * l_anchor + w_off * l_off + w_iou * l_iou
    return total, {'anchor': float(l_anchor), 'off': float(l_off),
                   'iou': float(l_iou), 'n': int(valid.sum())}


def decode_boundary(anchor_logits: torch.Tensor, dl_pred: torch.Tensor,
                    dr_pred: torch.Tensor):
    """-> (x_hat, x_left, x_right), each (N,) in column units.

    The anchor is the argmax, NOT an expectation: the anchor distribution is
    deliberately flat over the span (and can be multi-modal across a repeated
    passage), so its mean is meaningless.
    """
    w = anchor_logits.shape[1]
    a = anchor_logits.argmax(dim=1)
    idx = a[:, None]
    dl = torch.gather(dl_pred, 1, idx)[:, 0] * w
    dr = torch.gather(dr_pred, 1, idx)[:, 0] * w
    xl = (a.float() - dl).clamp(0, w - 1)
    xr = (a.float() + dr).clamp(0, w - 1)
    return 0.5 * (xl + xr), xl, xr


def decode_mask(anchor_logits: torch.Tensor, dl_pred: torch.Tensor,
                dr_pred: torch.Tensor, height: int, staff_row=None,
                halfwidth: int = 2):
    """Mask the UNMODIFIED metric code can consume (see bucketed_softmax for the
    two ways this went wrong before).

    staff_row: optional (N,) row for the y coordinate. Without it the bar spans
    the full height, which puts the centre of mass at H/2 -- fine on single-staff
    strips, WRONG on multi-staff pages, where it made every prediction land on
    the middle staff (room 10.6). Pass the coarse staff head's output here.
    """
    x_hat, _, _ = decode_boundary(anchor_logits, dl_pred, dr_pred)
    n, w = anchor_logits.shape
    dev = anchor_logits.device
    cx = x_hat.round().long().clamp(0, w - 1)
    cols = torch.arange(w, device=dev)[None, None, :]
    in_x = (cols - cx[:, None, None]).abs() <= halfwidth          # (N,1,W)
    rows = torch.arange(height, device=dev)[None, :, None]
    if staff_row is None:
        in_y = torch.ones((n, height, 1), dtype=torch.bool, device=dev)
    else:
        cy = staff_row.round().long().clamp(0, height - 1)
        in_y = (rows - cy[:, None, None]).abs() <= halfwidth      # (N,H,1)
    return (in_x & in_y).float().unsqueeze(1)
