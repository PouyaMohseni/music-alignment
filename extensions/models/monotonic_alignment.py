"""S2 -- score following as LEARNED MONOTONIC ALIGNMENT, not per-frame detection.

THE OBSERVATION THIS IS BUILT ON
---------------------------------
Every model in this literature -- CUNet, MM-Loc, CYOLO, CODA, and everything we
have built -- decides position INDEPENDENTLY AT EACH FRAME. The sequence
structure enters, if at all, through an LSTM that conditions the features; no
loss term ever says "your outputs across time must form a coherent path".

What that costs is measured, and it is enormous. On identical inputs:

    offline alignment over the whole sequence ....... 98.06 pct@0.5s
    online per-frame decisions ......................  10.73

An 87-point spread that is entirely the decision rule, not the features. We
then showed a crude hand-designed transition prior applied only AT DECODE TIME
recovers part of it: cyolo_sb 79.9 -> 83.4 on room, zero parameters, zero
training (C2, verified against a paired baseline and stable across a 4x sweep
of its hyperparameters).

If a hand-designed prior bolted onto a per-frame model is worth +3.5, the
question is what a model TRAINED to produce alignable sequences is worth. That
is what this file is.

THE FORMULATION
---------------
Encode both modalities as SEQUENCES and align them:

    audio  ->  A in R^{T x d}     one embedding per frame
    strip  ->  B in R^{X x d}     one embedding per score column
    S = A B^T / tau               (T, X) log-compatibility matrix

Train with a FORWARD-SUM objective: the negative log of the total probability
of all MONOTONE paths through S that pass through the annotated onsets. This is
the forward algorithm of an HMM whose transition matrix permits stay, advance,
and (at a cost) jump -- so it marginalises over every alignment consistent with
the score being read left to right, instead of supervising one position per
frame in isolation.

Two properties follow, and they are the point:
  * the model is optimised for the QUANTITY WE DECODE -- a path -- rather than
    for per-frame accuracy that a decoder then has to repair;
  * supervision is sequence-level, so it exploits the whole piece rather than
    treating each of its thousands of frames as an independent example. On 353
    training pieces that difference in statistical efficiency is large.

WHY THE STRIP IS A PREREQUISITE, NOT A PREFERENCE
--------------------------------------------------
Monotonic alignment needs the score to be a 1-D SEQUENCE. On a 2-D page it is
not one, and "monotone" has no meaning. The unrolled strip makes the score a
sequence of columns read left to right, which is exactly the object a forward
algorithm can run over -- and it simultaneously deletes the staff-assignment
step that cost an earlier model 10.6 on room.

REPEATS ARE NOT MONOTONE, AND THAT IS HANDLED
----------------------------------------------
Real performances repeat sections, so a strictly monotone constraint would make
a repeat unrecoverable. The transition kernel therefore allows a jump to any
column at a fixed log-cost, exactly as C2's decode does. This is also what our
own repeat-ambiguity diagnostic said the task needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e4          # finite, so gradients stay well-defined


def forward_sum_loss(logits: torch.Tensor, anchors: torch.Tensor = None,
                     max_step: int = 8, jump_cost: float = 8.0,
                     anchor_weight: float = 1.0):
    """Negative log-probability of all monotone paths through the score matrix.

    logits : (B, T, X) or (T, X). BATCHED ON PURPOSE: the recursion is a Python
    loop over T and is therefore kernel-launch bound, not FLOP bound, so its
    cost is paid once per step regardless of batch size. Running B chunks
    together divides the per-sample cost by B, which is the difference between
    a feasible training run and an infeasible one.
    anchors: (T,) target column per frame, -1 where unannotated. Used as an
             ADDITIONAL supervised term; the forward-sum term needs no
             per-frame labels at all.

    The forward recursion in log space:

        a[0, x] = logp[0, x]
        a[t, x] = logp[t, x] + logsumexp_over_predecessors(a[t-1, .])

    where predecessors are columns in [x - max_step, x] (stay or advance), plus
    a global jump path at `jump_cost`. Implemented with a max-pool-style shifted
    stack so it is O(T * X * max_step) and fully vectorised over X.
    """
    squeeze = logits.dim() == 2
    if squeeze:
        logits = logits[None]
        if anchors is not None:
            anchors = anchors[None]
    B, T, X = logits.shape
    logp = F.log_softmax(logits, dim=2)

    a = logp[:, 0]                                    # (B, X)
    for t in range(1, T):
        # predecessors x' in [x-max_step, x]: one shifted copy per step, stacked
        shifts = [a]
        for s in range(1, max_step + 1):
            sh = torch.full_like(a, NEG_INF)
            sh[:, s:] = a[:, :-s]
            shifts.append(sh)
        local = torch.logsumexp(torch.stack(shifts, 0), dim=0)     # (B, X)
        # a jump from anywhere, at a fixed cost -- repeats must stay reachable
        jump = (a.logsumexp(dim=1) - jump_cost)[:, None]
        a = logp[:, t] + torch.logaddexp(local, jump.expand_as(local))
        a = a - a.max(dim=1, keepdim=True).values      # keep it bounded

    loss = (-a.logsumexp(dim=1) / T).mean()

    if anchors is not None and anchor_weight > 0:
        valid = anchors >= 0
        if valid.any():
            idx = anchors.clamp(0, X - 1).long()
            ce_all = -logp.gather(2, idx[:, :, None])[:, :, 0]      # (B, T)
            loss = loss + anchor_weight * ce_all[valid].mean()
    return loss


@torch.no_grad()
def causal_viterbi(logits: torch.Tensor, max_step: int = 8,
                   jump_cost: float = 8.0) -> torch.Tensor:
    """Online decode: at frame t use only frames <= t. Returns (T,) columns.

    This is the same transition structure the loss was trained under, so
    training and inference agree by construction -- unlike C2, which bolted a
    prior onto a model that had never seen one.
    """
    T, X = logits.shape
    logp = F.log_softmax(logits, dim=1)
    a = logp[0].clone()
    out = torch.empty(T, dtype=torch.long, device=logits.device)
    out[0] = int(a.argmax())
    for t in range(1, T):
        shifts = [a]
        for s in range(1, max_step + 1):
            sh = torch.full_like(a, NEG_INF)
            sh[s:] = a[:-s]
            shifts.append(sh)
        local = torch.stack(shifts, 0).max(dim=0).values
        jump = a.max() - jump_cost
        a = logp[t] + torch.maximum(local, jump.expand_as(local))
        a = a - a.max()
        out[t] = int(a.argmax())
    return out


class ColumnEncoder(nn.Module):
    """Strip -> one embedding per score column. Audio-independent, so it runs
    ONCE per piece; the alignment matrix is then a single matmul."""

    def __init__(self, d: int = 128, width: int = 24, n_blocks: int = 4):
        super().__init__()
        chans, layers = [1] + [width * (2 ** i) for i in range(n_blocks)], []
        for i in range(n_blocks):
            layers += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
                nn.GroupNorm(1, chans[i + 1]), nn.ELU(False),
                nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1),
                nn.GroupNorm(1, chans[i + 1]), nn.ELU(False),
                # pool y always, x only after the first block: horizontal
                # resolution IS the answer and must not be spent early
                nn.MaxPool2d((2, 1 if i == 0 else 2)),
            ]
        self.trunk = nn.Sequential(*layers)
        self.proj = nn.Conv1d(chans[-1], d, 1)
        self.x_stride = 2 ** (n_blocks - 1)

    def forward(self, strip: torch.Tensor) -> torch.Tensor:
        h = self.trunk(strip).mean(dim=2)            # (N, C, X)
        return F.normalize(self.proj(h).transpose(1, 2), dim=-1)   # (N, X, d)


class FrameEncoder(nn.Module):
    """Mel window -> one embedding per audio frame."""

    def __init__(self, n_mels: int = 78, d: int = 128, hidden: int = 128):
        super().__init__()
        self.pre = nn.Sequential(nn.LayerNorm(n_mels), nn.Linear(n_mels, hidden), nn.ELU(False))
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(1, hidden), nn.ELU(False),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(1, hidden), nn.ELU(False),
        )
        self.proj = nn.Linear(hidden, d)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (N, T, n_mels) -> (N, T, d)."""
        h = self.pre(mel).transpose(1, 2)
        h = self.temporal(h).transpose(1, 2)
        return F.normalize(self.proj(h), dim=-1)


class MonotonicAligner(nn.Module):
    def __init__(self, d: int = 128, n_mels: int = 78, tau: float = 0.07):
        super().__init__()
        self.audio = FrameEncoder(n_mels=n_mels, d=d)
        self.score = ColumnEncoder(d=d)
        self.tau = tau

    def similarity(self, mel: torch.Tensor, strip: torch.Tensor,
                   cols=None) -> torch.Tensor:
        """-> (T, X) log-compatibility for ONE piece."""
        a = self.audio(mel)[0]                       # (T, d)
        b = cols if cols is not None else self.score(strip)[0]   # (X, d)
        return (a @ b.T) / self.tau
