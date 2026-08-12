"""P1 -- bucketed-softmax position output, replacing the dense soft-Dice heatmap.

WHY THIS EXPERIMENT. The strongest architectural signal we have is a
within-paper, same-lab, same-data, same-audio-tower comparison in Frontiers
Table 4: MM-Loc scores 58.5 on MSMD-Rec `room` where CUNet -- the dense
soft-Dice heatmap family we build on -- scores 22.4. A 36-point swing whose
only material difference is the OUTPUT PARAMETERISATION: softmax classification
over position versus per-pixel Dice.

Two independent results point the same way:
  * AMT recovers 91-95% of onsets from the very room recordings our models
    score 56.6 on, and the room costs a transcriber ~0 F1 at 50 ms.  The
    information survives; the architecture drops it.
  * Real-IR augmentation bought CYOLO (a DETECTION model) +25.2 on room but
    bought our heatmap model only +11 from the same IR bank.

Soft-Dice is also known to win in-distribution and lose out-of-distribution
(Galdran et al.), and synth->real is exactly that shift.

ZERO NEW PARAMETERS. `conv_out` is nn.Conv2d(n, 1, 1x1), so its output is
already a per-pixel logit map.  We softmax it over the whole page and train it
as a distribution over position.  No layer is added or resized, the checkpoint
stays shape-compatible with its warm start, and the run differs from
R2r_realir in LOSS AND DECODE ONLY.

WHY 2-D, AND WHY THE FIRST VERSION FAILED (room 10.6)
-----------------------------------------------------
The first implementation marginalised height away and classified over x
columns only, on the assumption -- taken from a CLAUDE.md note -- that the
strips are single-staff so y carries no information.  That note describes the
`cpjku_adapter` path.  The CPJKU-NATIVE data these experiments actually train
and evaluate on is full multi-staff pages: measured on
msmd_real_performances/score, 5-6 distinct staff rows per page with y spanning
125..1068.

That matters because calculate_batch_stats uses y to decide WHICH STAFF the
prediction is on, and then unrolls x against that staff's cumulative offset:

    staff_id_pred = argmin |staff_coords - com_pred[0]|
    x_coord_pred  = com_pred[1] + add_per_staff[staff_id_pred]

A height-marginalised prediction has centre of mass at y = H/2 for every
frame, so it always selects the middle staff, and every x is unrolled against
the wrong offset.  The x was accurate; the staff was not.  Hence 10.6 on room
while the training loss was converging normally.

So position is classified over the FULL H x W grid, which is what "bucketed
softmax over position" should have meant on a page in the first place.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _flat_logits(logits: torch.Tensor) -> torch.Tensor:
    """(N,1,H,W) -> (N, H*W)."""
    if logits.dim() != 4:
        raise ValueError(f'expected (N,1,H,W) logits, got {tuple(logits.shape)}')
    n = logits.shape[0]
    return logits[:, 0].reshape(n, -1)


def target_distribution(y: torch.Tensor, eps: float = 1e-8):
    """(N,1,H,W) GT mask -> ((N, H*W) normalised target, (N,) valid mask).

    Frames whose mask is entirely empty (target off the page) would divide by
    zero.  They are returned as all-zero rows and excluded via `valid`, rather
    than becoming a uniform target -- which would actively train the model to
    spread probability across the whole page.
    """
    n = y.shape[0]
    m = y[:, 0].reshape(n, -1)
    total = m.sum(dim=1, keepdim=True)
    valid = (total.squeeze(1) > eps)
    q = torch.where(total > eps, m / total.clamp_min(eps), torch.zeros_like(m))
    return q, valid


def bucketed_ce_loss(logits: torch.Tensor, y: torch.Tensor, pool: str = 'logsumexp'):
    """Soft cross-entropy between a softmax over the page and the GT mask.

    Soft targets, not argmax: the GT blob has finite width that encodes
    sub-pixel position and the tolerance the task actually cares about, so
    hardening it would make the objective stricter than the metric.  Reduces to
    ordinary cross-entropy when the target is one-hot.  `pool` is accepted and
    ignored, kept so callers and configs need not change.

    Returns (loss, n_valid), averaged over valid frames.
    """
    logp = F.log_softmax(_flat_logits(logits), dim=1)
    q, valid = target_distribution(y)
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0          # keeps the graph alive
    ce = -(q[valid] * logp[valid]).sum(dim=1)
    return ce.mean(), int(valid.sum().item())


def decode_position(logits: torch.Tensor, pool: str = 'logsumexp', refine: int = 2):
    """(N,1,H,W) -> (y_hat, x_hat), each (N,) float.

    argmax over the page, then a local expectation in a (2*refine+1)^2 window
    for sub-pixel accuracy.  NOT a global expectation: the distribution is
    routinely multi-modal (the same passage recurs at several places on the
    page), and the mean of two modes lands in the gap between them.
    """
    n, _, h, w = logits.shape
    p = F.softmax(_flat_logits(logits), dim=1)                 # (N, H*W)
    idx = p.argmax(dim=1)
    y0 = torch.div(idx, w, rounding_mode='floor')
    x0 = idx % w
    if refine <= 0:
        return y0.float(), x0.float()

    k = 2 * refine + 1
    offs = torch.arange(-refine, refine + 1, device=logits.device)
    dy = offs.repeat_interleave(k)                             # (k*k,)
    dx = offs.repeat(k)                                        # (k*k,)
    yy = (y0[:, None] + dy[None, :]).clamp(0, h - 1)           # (N, k*k)
    xx = (x0[:, None] + dx[None, :]).clamp(0, w - 1)
    wts = torch.gather(p, 1, yy * w + xx)                      # (N, k*k)
    den = wts.sum(1).clamp_min(1e-12)
    y_hat = (wts * yy.float()).sum(1) / den
    x_hat = (wts * xx.float()).sum(1) / den
    return y_hat, x_hat


def decode_mask(logits: torch.Tensor, height: int, pool: str = 'logsumexp',
                halfwidth: int = 2, refine: int = 2) -> torch.Tensor:
    """Build a mask the UNMODIFIED metric code can consume.

    calculate_batch_stats binarises at 0.5 and takes the centre of mass of what
    survives, then uses that COM's y to pick a staff and its x to unroll.  Two
    approaches already failed here and both are worth remembering:

      1. Broadcast the peak-normalised softmax and let COM find the centre.
         A diffuse or multi-modal map leaves a huge above-half-maximum plateau
         (measured: ~13x wider than the target) whose centre of mass sits
         nowhere near the peak.
      2. Emit a full-height bar at the decoded column.  Correct in x, but COM
         lands at y = H/2 on every frame, so the metric assigns every
         prediction to the middle staff of a multi-staff page and unrolls x
         against the wrong offset.  This is what scored 10.6 on room.

    So we decode (y, x) ourselves and hand the metric a small square centred
    there.  Centre of mass is then exactly (round(y_hat), round(x_hat)) by
    construction, which decouples "how we read a position out of the
    distribution" from "how the metric reads a position out of a mask".
    Everything downstream -- staff assignment, unrolling, interpol_c2o, the
    threshold sweep -- stays byte-identical to every other experiment.

    Rounding costs at most half a pixel, far below the 0.5 s tolerance.
    """
    n, _, h, w = logits.shape
    y_hat, x_hat = decode_position(logits, pool=pool, refine=refine)
    cy = y_hat.round().long().clamp(0, h - 1)
    cx = x_hat.round().long().clamp(0, w - 1)
    rows = torch.arange(h, device=logits.device)[None, :, None]      # (1,H,1)
    cols = torch.arange(w, device=logits.device)[None, None, :]      # (1,1,W)
    blob = (((rows - cy[:, None, None]).abs() <= halfwidth) &
            ((cols - cx[:, None, None]).abs() <= halfwidth)).float()  # (N,H,W)
    return blob[:, None].contiguous()
