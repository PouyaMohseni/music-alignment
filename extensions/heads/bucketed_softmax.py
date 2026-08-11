"""P1 -- bucketed-softmax position output, replacing the dense soft-Dice heatmap.

WHY THIS EXPERIMENT. The single strongest architectural signal we have is a
within-paper, same-lab, same-data, same-audio-tower comparison in Frontiers
Table 4: MM-Loc scores 58.5 on MSMD-Rec `room` where CUNet -- the dense
soft-Dice heatmap family we build on -- scores 22.4. A 36-point swing whose
only material difference is the OUTPUT PARAMETERISATION: bucketed softmax
classification over position versus per-pixel Dice.

Two independent results point the same way:
  * AMT recovers 91-95% of onsets from the very room recordings our models
    score 45.6 on, and the room costs a transcriber ~0 F1.  The information
    survives; our architecture drops it.  The failure is representational.
  * Real-IR augmentation bought CYOLO (a DETECTION model) +25.2 on room but
    bought our heatmap model only +11 (45.6 -> 56.6) from the same IR bank.
    The intervention transfers only partly, which is what you would expect if
    the output layer -- not the input distribution -- is the binding
    constraint.

There is also external support: soft-Dice is known to win in-distribution and
lose out-of-distribution (Galdran et al.), and synth->real is exactly an
out-of-distribution shift.

ZERO NEW PARAMETERS -- WHY THAT MATTERS. `conv_out` is nn.Conv2d(n, 1, 1x1), so
its output is already a per-pixel logit map.  We marginalise those logits over
height to get one logit per x column and softmax over x.  No layer is added,
no weight is resized, and the checkpoint stays shape-compatible with the base
model.  So this run differs from its warm start in the LOSS AND DECODE ONLY,
which is the cleanest possible isolation of the variable under test -- and it
means a regression cannot be blamed on a new randomly-initialised head.

WHY MARGINALISING OVER HEIGHT IS LOSSLESS HERE. Our strips are single-staff
(CLAUDE.md: add_per_staff is always [[H//2], [0]]), so calculate_batch_stats
maps every prediction to the one staff row and position is carried entirely by
x.  Collapsing y therefore discards nothing the metric reads.

LOGSUMEXP, NOT MEAN. Marginalising a joint log-density over a nuisance
dimension is logsumexp, not averaging: it is a soft-OR ("the position is here
if ANY row fires here"), whereas the mean is a soft-AND that a single strong
row cannot carry.  Mean pooling is available via pool='mean' for ablation.

SOFT TARGETS, NOT ARGMAX. The GT mask is a blob with finite width, and its
x-marginal encodes sub-pixel position plus the tolerance the task actually
cares about.  Taking argmax would throw that away and make the target harsher
than the metric.  We use the normalised x-marginal as a soft target, which
degenerates to ordinary cross-entropy when the target is one-hot.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def marginalise_x(logits: torch.Tensor, pool: str = 'logsumexp') -> torch.Tensor:
    """(N, 1, H, W) per-pixel logits -> (N, W) per-column logits."""
    if logits.dim() != 4:
        raise ValueError(f'expected (N,1,H,W) logits, got {tuple(logits.shape)}')
    z = logits[:, 0]                                  # (N, H, W)
    if pool == 'logsumexp':
        return torch.logsumexp(z, dim=1)              # (N, W)
    if pool == 'mean':
        return z.mean(dim=1)
    if pool == 'max':
        return z.max(dim=1).values
    raise ValueError(f'unknown pool {pool!r}')


def target_x_distribution(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(N, 1, H, W) GT mask -> (N, W) normalised x-marginal.

    Frames whose mask is entirely empty (the target has left the strip) would
    divide by zero; they are returned as all-zero rows and masked out of the
    loss by `valid` rather than silently becoming a uniform target, which would
    actively teach the model to spread probability everywhere.
    """
    m = y[:, 0].sum(dim=1)                            # (N, W), sum over height
    total = m.sum(dim=1, keepdim=True)
    valid = (total.squeeze(1) > eps)
    p = torch.where(total > eps, m / total.clamp_min(eps), torch.zeros_like(m))
    return p, valid


def bucketed_ce_loss(logits: torch.Tensor, y: torch.Tensor,
                     pool: str = 'logsumexp') -> tuple:
    """Soft cross-entropy between softmax over x and the GT x-marginal.

    Returns (loss, n_valid).  Averaged over valid frames only.
    """
    x_logits = marginalise_x(logits, pool)            # (N, W)
    logp = F.log_softmax(x_logits, dim=1)
    q, valid = target_x_distribution(y)
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0                  # keeps the graph alive
    ce = -(q[valid] * logp[valid]).sum(dim=1)         # (n_valid,)
    return ce.mean(), int(valid.sum().item())


def decode_position(logits: torch.Tensor, pool: str = 'logsumexp',
                    refine: int = 3) -> torch.Tensor:
    """(N,1,H,W) logits -> (N,) decoded x position, in columns.

    argmax, then a local expectation over +-`refine` columns for sub-pixel
    accuracy.  NOT a global expectation: the distribution is routinely
    multi-modal (the same passage appears at several x on the strip), and the
    mean of two modes lands in the gap between them, which is precisely the
    failure that made the first P1 run score 10.6 on room.
    """
    x_logits = marginalise_x(logits, pool)                       # (N, W)
    p = F.softmax(x_logits, dim=1)
    n, w = p.shape
    idx = p.argmax(dim=1)                                        # (N,)
    if refine <= 0:
        return idx.float()
    offs = torch.arange(-refine, refine + 1, device=p.device)
    cols = (idx[:, None] + offs[None, :]).clamp(0, w - 1)        # (N, 2r+1)
    wts = torch.gather(p, 1, cols)
    return (wts * cols.float()).sum(1) / wts.sum(1).clamp_min(1e-12)


def decode_mask(logits: torch.Tensor, height: int,
                pool: str = 'logsumexp', halfwidth: int = 2,
                refine: int = 3) -> torch.Tensor:
    """Build a mask the UNMODIFIED metric code can consume.

    calculate_batch_stats binarises at 0.5 and then takes the centre of mass of
    whatever survives.  The obvious approach -- broadcast the peak-normalised
    softmax and let it find the centre -- FAILS, and cost a full training run:
    a diffuse or multi-modal distribution leaves a huge above-half-maximum
    plateau (measured: ~13x wider than the target, precision 0.076), and its
    centre of mass sits nowhere near the peak.

    So we decode the position ourselves and hand the metric an unambiguous
    delta: a symmetric bar of width 2*halfwidth+1 centred on the decoded
    column, spanning the full height.  Centre of mass is then exactly
    (H/2, round(x_hat)) by construction, which decouples "how we read a
    position out of the distribution" from "how the metric reads a position out
    of a mask".  Everything downstream -- unrolling, interpol_c2o, the
    threshold sweep -- stays byte-identical to every other experiment.

    Rounding to a whole column costs at most 0.5 px, which at w_scale=4 is 2
    original pixels -- far below the 0.5 s tolerance the metric applies.
    """
    x_hat = decode_position(logits, pool=pool, refine=refine)    # (N,)
    n = x_hat.shape[0]
    w = logits.shape[-1]
    centre = x_hat.round().long().clamp(0, w - 1)
    cols = torch.arange(w, device=logits.device)[None, :]        # (1, W)
    bar = ((cols - centre[:, None]).abs() <= halfwidth).float()  # (N, W)
    return bar[:, None, None, :].expand(-1, 1, height, -1).contiguous()
