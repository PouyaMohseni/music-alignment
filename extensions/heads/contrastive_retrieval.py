"""A2 -- position as cross-modal RETRIEVAL, then temporal filtering.

WHY A RANKED OUTPUT
-------------------
The single largest controlled effect in this literature is the output
parameterisation: MM-Loc scores 58.5 on real audio where CUNet -- same paper,
same data, same audio tower -- scores 22.4, and the difference is bucketed
softmax classification versus a dense soft-Dice heatmap. Our own numbers agree
from a different direction: the detection formulation degrades -9.4 from
synthetic to real where our heatmap degrades ~-30.

What detection and bucketed-softmax share, and what soft-Dice lacks, is that
the decision is RANKED. A ranking only has to get the ordering right; a dense
per-pixel regression has to get calibration right too, and calibration is
exactly what a domain shift destroys. Retrieval is the purest form of a ranked
decision: embed the audio, embed every candidate score location, and take the
best match.

FORMULATION
-----------
    audio at time t   -> a_t in R^d   (unit norm)
    score column x    -> s_x in R^d   (unit norm)
    similarity        -> S[t, x] = a_t . s_x / tau

Trained with InfoNCE over columns: the positive is the true column at t, the
negatives are all other columns of the SAME page. Same-page negatives are what
force the model to discriminate within a piece rather than merely telling pieces
apart -- with in-batch cross-piece negatives the task collapses to piece
identification, which is trivially solvable and teaches nothing about position.

Inference does not argmax S per frame. A per-frame argmax is exactly the
"teleport" failure we have diagnosed repeatedly: the similarity profile of a
piece with a repeated passage is genuinely multi-modal, and an independent
per-frame decision hops between modes. Instead S is treated as a per-frame
log-likelihood over position and filtered with a causal monotone-ish transition
prior (`filter_similarity`), so the sequence decision is made once, with
continuity, rather than T times independently.

NO SYMBOLIC INTERMEDIATE. The score side is a learned encoder over image
columns. Nothing is transcribed, no note events, no MIDI.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnScoreEncoder(nn.Module):
    """Score image -> one embedding per x column: (N,1,H,W) -> (N, W', d)."""

    def __init__(self, d: int = 128, width: int = 64, n_down: int = 3):
        super().__init__()
        layers, c_in = [], 1
        for i in range(n_down):
            c_out = width * (2 ** i)
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GroupNorm(1, c_out), nn.ELU(False),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GroupNorm(1, c_out), nn.ELU(False),
                # pool y twice as fast as x: we must not throw away horizontal
                # resolution, which IS the quantity being predicted
                nn.MaxPool2d((2, 2) if i < n_down - 1 else (2, 1)),
            ]
            c_in = c_out
        self.trunk = nn.Sequential(*layers)
        self.proj = nn.Conv1d(c_in, d, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)                     # (N, C, H', W')
        h = h.mean(dim=2)                     # collapse y -> (N, C, W')
        h = self.proj(h).transpose(1, 2)      # (N, W', d)
        return F.normalize(h, dim=-1)


class AudioQueryEncoder(nn.Module):
    """Audio feature window -> one query embedding: (N, T, F) -> (N, d)."""

    def __init__(self, in_dim: int, d: int = 128, hidden: int = 256):
        super().__init__()
        self.pre = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.ELU(False))
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GroupNorm(1, hidden), nn.ELU(False),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GroupNorm(1, hidden), nn.ELU(False),
        )
        self.proj = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x).transpose(1, 2)        # (N, hidden, T)
        h = self.temporal(h).mean(dim=2)       # (N, hidden)
        return F.normalize(self.proj(h), dim=-1)


def infonce_over_columns(a: torch.Tensor, s: torch.Tensor, target_col: torch.Tensor,
                         tau: float = 0.07, sigma: float = 2.0):
    """InfoNCE with SAME-PAGE column negatives and a soft positive band.

    a: (N, d) audio queries; s: (N, W', d) column embeddings for that same page;
    target_col: (N,) true column index in W' units.

    The positive is a small Gaussian band rather than a single column: columns
    are ~sub-notehead wide after downsampling, so declaring the neighbours to be
    hard negatives would teach the model that a one-column error is as bad as
    landing on the wrong system, which the metric does not say.
    """
    logits = torch.einsum('nd,nwd->nw', a, s) / tau        # (N, W')
    w = logits.shape[1]
    cols = torch.arange(w, device=a.device).float()[None, :]
    band = torch.exp(-((cols - target_col[:, None].float()) ** 2) / (2 * sigma ** 2))
    q = band / band.sum(dim=1, keepdim=True).clamp_min(1e-12)
    loss = -(q * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    with torch.no_grad():
        acc = (logits.argmax(1) - target_col).abs().float().mean()
    return loss, logits, acc


@torch.no_grad()
def filter_similarity(sim: torch.Tensor, max_step: int = 6, jump_cost: float = 8.0,
                      stay_bonus: float = 0.0):
    """Causal Viterbi-style filter over a (T, W) similarity matrix -> (T,) cols.

    Per-frame argmax is the teleport failure: with a repeated passage the
    profile is multi-modal and independent decisions hop between modes. This
    accumulates a score with a transition penalty, so moving forward by up to
    `max_step` columns is cheap and jumping anywhere else costs `jump_cost` --
    which permits a genuine repeat jump but makes it pay for itself.

    Causal: frame t uses only frames <= t. Returns the running argmax, i.e.
    what an online system could have emitted at each frame, NOT a backtraced
    global path (that would be non-causal and not a score-following result).
    """
    T, W = sim.shape
    acc = sim[0].clone()
    out = torch.empty(T, dtype=torch.long, device=sim.device)
    out[0] = int(acc.argmax())
    for t in range(1, T):
        # best predecessor within the forward window, via a max-pool over a
        # left-aligned band; anywhere else pays jump_cost
        pad = F.pad(acc[None, None, :], (max_step, 0), value=float('-inf'))
        local = F.max_pool1d(pad, kernel_size=max_step + 1, stride=1)[0, 0]  # (W,)
        best_jump = acc.max() - jump_cost
        acc = torch.maximum(local + stay_bonus, best_jump) + sim[t]
        acc = acc - acc.max()                  # keep numerically bounded
        out[t] = int(acc.argmax())
    return out
