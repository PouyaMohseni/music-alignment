"""C1 -- distil the AMT posteriorgram into CYOLO's audio encoder.

THE MEASUREMENT THIS IS BUILT ON
--------------------------------
Two microphones on the SAME performances (results/amt_bridge_eval-437566.log):
room reverberation costs a piano AMT model 0.001 onset F1 (0.9116 room vs
0.9124 direct pickup at 50 ms), while our trackers lose ~30 points on those
same files. The AMT network's internal representation is room-invariant in a
way our audio encoders are not.

WHY DISTIL IT RATHER THAN FEED IT
---------------------------------
We already tried feeding borrowed representations as INPUT. Every time, room
accuracy fell as input dimension rose:

    native mel  78-dim -> 67.1 | MERT 768 -> 56.6 | +xattn -> 35.3 / 19.3 / 2.6

and the MERT-in-detector run overfit 12.6x (train frame-diff 2.04, val 25.73)
on 353 training pieces. Adding test-time capacity is the one direction our own
evidence has repeatedly contradicted.

Distillation inverts that. The posteriorgram becomes a TRAINING TARGET, not an
input:
  * inference capacity is UNCHANGED -- the encoder stays the native 78-band mel
    CNN, and this head is deleted at test time;
  * what transfers is the room-invariance, as a constraint on what the
    conditioning vector z must encode;
  * it is extra supervision on a fixed-size model, which is exactly the useful
    direction when data is the binding constraint.

Closest measured precedent: our pitch auxiliary loss was worth +5.2 on room
over the same base (B1a 38.5 -> B2 43.7), with the head deleted at inference.
This is that idea with a target that additionally carries room-invariance.

WHAT z HAS TO PREDICT
---------------------
CYOLO's ContextConditioning turns a 40-frame window into one conditioning
vector z (zdim=128) per frame position. The target is the posteriorgram row at
that SAME frame -- 176 dims = 88 frame-activity + 88 onset-regression. So z is
constrained to encode what is sounding right now, which is precisely the
quantity the position decision depends on and precisely what the room does not
destroy.

The frame grids already match by construction: scripts/precompute_amt_posteriorgram.py
resampled to CYOLO's own frame count at its true 20.0091 fps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PosteriorgramDistillHead(nn.Module):
    """z -> predicted posteriorgram at the current frame. Deleted at inference.

    Deliberately small (one hidden layer): its job is to make z informative,
    not to be a good transcriber itself. A large head would let the head do the
    work and leave z unconstrained -- the standard failure mode of auxiliary
    losses.
    """

    def __init__(self, zdim: int = 128, out_dim: int = 176, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(False),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)                     # logits, (N, out_dim)


def distill_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor = None):
    """BCE against the soft posteriorgram, which is already in [0, 1].

    Soft targets, not thresholded: the whole point is to transfer the
    transcriber's CALIBRATED uncertainty. Hardening to a binary piano roll
    would discard exactly the information that makes a distillation target
    better than a label -- and would also discard the room-robustness, which
    shows up as graceful degradation of the posteriors rather than as different
    argmaxes.

    Returns (loss, n_valid).
    """
    if valid is not None:
        if valid.sum() == 0:
            return logits.sum() * 0.0, 0
        logits, target = logits[valid], target[valid]
    t = target.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(logits, t), int(logits.shape[0])
