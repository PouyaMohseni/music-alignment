"""S1 -- a small 1-D conditioned localiser over the UNROLLED strip.

WHY A STRIP, AND WHY NOT YOLO
-----------------------------
On a 2-D page, position is (x, y), and the metric uses y to choose a staff and
then unrolls x against THAT staff's cumulative offset. A perfectly correct x on
the wrong staff scores about zero -- we measured it: P1 scored 10.6 on room for
exactly that reason, and a whole coarse-staff head had to be built to patch it.

On an unrolled strip the staff-assignment step DOES NOT EXIST. Position is one
number, interpol_c2o(x) is direct, and that failure mode is deleted by
construction rather than mitigated. Monotonicity also becomes trivially
expressible: forward means x increases.

Given that, a 2-D detector is over-engineered. With one object to find per frame
in 1-D there is no need for NMS, for three anchors at three scales, for 2-D box
regression, or for a class head whose "system" class is the entire image.
Carrying that machinery is the same capacity mistake our own numbers have
punished five times on 353 training pieces (768-dim MERT 56.6, +cross-attention
35.3 / 19.3, +DINOv2 2.6, and a 12.6x train/val overfit).

WHAT WE KEEP FROM THE DETECTION FAMILY, AND WHY
-----------------------------------------------
Only the property the evidence actually supports: the output is a RANKED
decision, not a calibrated dense map. A ranking needs the ordering right; a
per-pixel probability map needs calibration too, and calibration is what a
synthetic->real shift destroys (soft-Dice is known to win in-distribution and
lose out-of-distribution). Objectness-per-column is a ranking. Soft-Dice is not.

WHAT THE EVIDENCE SUPPORTS, CHECKED
------------------------------------
Frontiers Table 4 is a single table, one paper, one protocol:

    dense heatmap 22.4  <<  bucketed softmax 58.5  <<  detection 71.2  <  +sb 79.9

So within one protocol, detection beats a bucketed softmax by 12.7 on room, and
both crush the dense heatmap. An earlier version of this file argued the
opposite by comparing MM-Loc's 58.5 (Frontiers) against CYOLO-without-IR's 46.0
(Henkel & Widmer, EUSIPCO 2021) -- two different papers, which is precisely the
protocol-mixing we criticise elsewhere. Corrected.

Hence this model keeps BOTH properties that separate detection from the
alternatives: the output is RANKED (ordering, not calibration -- what survives a
domain shift), and it predicts EXTENT, not just a point. What it drops is only
the 2-D machinery that a strip makes meaningless: NMS, multi-scale anchors, 2-D
box regression, and a "system" class that would cover the entire image.

DESIGN
------
    strip (N,1,H,W) --conv--> collapse y --> (N,C,W')   one feature per column
    z (audio)       --FiLM--> modulates the trunk
    head            --> objectness logit per column + sub-column offset

FiLM because it is the only conditioning that has ever worked here: our own
cross-attention variants scored 19.3 and 2.6 on room against FiLM's 38.5, and a
seven-way controlled ablation in the literature found FiLM, AdaLN, adaLN-Zero,
cross-attention, prefix and additive injection all comparable.

The offset head exists because a column at stride 16 (half-scale) covers ~32
original pixels; without it the quantisation floor alone would dominate the
tight-tolerance columns we are weakest on (0.05 s / 0.1 s).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM1d(nn.Module):
    """Per-channel scale/shift from the audio vector, applied to a 2-D map."""

    def __init__(self, zdim: int, channels: int):
        super().__init__()
        self.gamma = nn.Linear(zdim, channels)
        self.beta = nn.Linear(zdim, channels)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        g = self.gamma(z)[:, :, None, None]
        b = self.beta(z)[:, :, None, None]
        return x * g + b


class StripBlock(nn.Module):
    """Conv pair, optional FiLM, then pool. Pools y FASTER than x: the answer
    is horizontal, so vertical resolution is spendable and horizontal is not."""

    def __init__(self, c_in: int, c_out: int, zdim: int, film: bool,
                 pool_x: int = 2, pool_y: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(1, c_out), nn.ELU(False),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(1, c_out), nn.ELU(False),
        )
        self.film = FiLM1d(zdim, c_out) if film else None
        self.pool = nn.MaxPool2d((pool_y, pool_x))

    def forward(self, x, z):
        x = self.conv(x)
        if self.film is not None:
            x = self.film(x, z)
        return self.pool(x)


class StripLocaliser(nn.Module):
    """(strip, z) -> (objectness logit per column, sub-column offset per column).

    Total downsample in x is 2**n_blocks_with_pool_x. With the default four
    blocks that is 16, so at half-scale input one column covers ~32 original
    pixels -- hence the offset head.
    """

    def __init__(self, zdim: int = 128, width: int = 24, n_blocks: int = 4,
                 film_from: int = 1, n_bar_classes: int = 0):
        super().__init__()
        chans = [1] + [width * (2 ** i) for i in range(n_blocks)]
        blocks = []
        for i in range(n_blocks):
            # keep x resolution in the FIRST block (pool_x=1): the earliest
            # layers are where notehead-scale detail lives
            blocks.append(StripBlock(chans[i], chans[i + 1], zdim,
                                     film=(i >= film_from),
                                     pool_x=(1 if i == 0 else 2),
                                     pool_y=2))
        self.blocks = nn.ModuleList(blocks)
        c = chans[-1]
        self.col = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1), nn.GroupNorm(1, c), nn.ELU(False))
        self.obj = nn.Conv1d(c, 1, 3, padding=1)
        self.off = nn.Conv1d(c, 1, 3, padding=1)
        # EXTENT, not just position. This is the one thing detection has over a
        # bucketed softmax, and within Frontiers Table 4 -- same paper, same
        # protocol -- detection (71.2) beats bucketed softmax (58.5) by 12.7 on
        # room. Predicting a width makes this a 1-D anchor-free detector rather
        # than a classifier over columns.
        self.wid = nn.Conv1d(c, 1, 3, padding=1)
        # auxiliary bar head: multi-granularity was worth +4.9 to cyolo_sb, and
        # "bar" still means something on a strip where "system" does not
        self.bar = nn.Conv1d(c, 1, 3, padding=1) if n_bar_classes else None
        self.x_stride = 2 ** (n_blocks - 1)

    def forward(self, strip: torch.Tensor, z: torch.Tensor):
        x = strip
        for b in self.blocks:
            x = b(x, z)
        x = x.mean(dim=2)                       # collapse y -> (N, C, W')
        x = self.col(x)
        obj = self.obj(x)[:, 0]                 # (N, W')
        off = torch.tanh(self.off(x)[:, 0])     # (N, W') in (-1, 1) columns
        wid = F.softplus(self.wid(x)[:, 0])     # (N, W'), >= 0, in columns
        bar = self.bar(x)[:, 0] if self.bar is not None else None
        return obj, off, wid, bar


def soft_target(x_true_col: torch.Tensor, n_cols: int, sigma: float = 1.5):
    """Gaussian band over columns, normalised. Soft rather than one-hot: a
    neighbouring column is nearly as good, and declaring it a hard negative
    would make the objective stricter than the metric."""
    cols = torch.arange(n_cols, device=x_true_col.device).float()[None, :]
    b = torch.exp(-((cols - x_true_col[:, None]) ** 2) / (2 * sigma ** 2))
    return b / b.sum(dim=1, keepdim=True).clamp_min(1e-12)


def localiser_loss(obj, off, x_true_col, wid=None, w_true_col=None,
                   valid=None, w_off: float = 5.0, w_wid: float = 1.0):
    """Ranked CE over columns + L1 on the sub-column offset at the true column.

    CE (not per-column BCE) is what makes this a RANKING: it only constrains the
    ordering of columns, which is the property that survives domain shift.
    """
    if valid is not None:
        if valid.sum() == 0:
            return obj.sum() * 0.0, {'ce': 0.0, 'off': 0.0, 'wid': 0.0, 'n': 0}
        obj, off, x_true_col = obj[valid], off[valid], x_true_col[valid]
        if wid is not None:
            wid = wid[valid]
        if w_true_col is not None:
            w_true_col = w_true_col[valid]

    q = soft_target(x_true_col, obj.shape[1])
    ce = -(q * F.log_softmax(obj, dim=1)).sum(dim=1).mean()

    idx = x_true_col.round().long().clamp(0, obj.shape[1] - 1)
    off_pred = torch.gather(off, 1, idx[:, None])[:, 0]
    off_true = (x_true_col - idx.float()).clamp(-1, 1)
    l_off = (off_pred - off_true).abs().mean()

    l_wid = obj.sum() * 0.0
    if wid is not None and w_true_col is not None:
        w_pred = torch.gather(wid, 1, idx[:, None])[:, 0]
        l_wid = (w_pred - w_true_col).abs().mean()

    total = ce + w_off * l_off + w_wid * l_wid
    return total, {'ce': float(ce), 'off': float(l_off), 'wid': float(l_wid),
                   'n': int(obj.shape[0])}


@torch.no_grad()
def decode(obj: torch.Tensor, off: torch.Tensor, x_stride: int = 16,
           scale: float = 2.0):
    """-> predicted x in ORIGINAL strip pixels.

    argmax, not expectation: a repeated passage makes the column distribution
    genuinely multi-modal, and the mean of two modes lands in the gap between
    them (the failure that produced 10.6 once already).
    """
    i = obj.argmax(dim=1)
    o = torch.gather(off, 1, i[:, None])[:, 0]
    return (i.float() + o) * x_stride * scale
