"""C2 -- causal temporal prior on cyolo_sb's decode. Zero parameters, zero retraining.

WHAT IT REPLACES
----------------
`cyolo_score_following/utils/general.py:get_max_box` decodes position as

    _, idx = x[..., 4].max(-1)      # argmax over objectness
    max_per_sample = x[idx][:4]

an INDEPENDENT argmax at every frame. Nothing links frame t to frame t-1. A
score page contains many visually similar passages, so the objectness map is
genuinely multi-modal, and an independent decision hops between modes -- the
"teleport" failure we have diagnosed repeatedly.

The evidence that this is where the loss is: on identical inputs, an offline
aligner scores 98.06 on room where our online greedy matcher scores 10.73 with
a median error of 8-13 seconds. Nearly all of that spread is the decision rule,
not the features.

WHY THIS IS THE CHEAPEST POSSIBLE INTERVENTION
-----------------------------------------------
It is a DECODE change. The released cyolo_sb checkpoint (79.9) is untouched, no
training runs, and the measurement costs one CPU eval. Our closest measured
precedent is the gated belief filter, worth +6.2 on room over the same base on
our own architecture (38.5 -> 44.7).

THE RULE
--------
Score each candidate box by its own objectness plus a transition term against
the previous accepted position:

    score(c) = log p_obj(c) + lambda * log P(c | prev)

with P a heavy-tailed transition prior over the unrolled x displacement:
forward motion at a plausible speed is nearly free, standing still is cheap,
and a large jump is expensive but NOT forbidden -- repeats and page turns are
real, and a hard monotonicity constraint would make them unrecoverable. This is
the same reason the A2 filter uses a jump cost rather than a monotone mask.

CAUSAL, and state is keyed per piece so it resets at piece boundaries and does
not depend on how the loader batches frames.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class TemporalDecoder:
    """Per-piece causal filter over per-frame detection candidates."""

    def __init__(self, lam: float = 1.0, fwd_px: float = 6.0, sigma_px: float = 18.0,
                 jump_logp: float = -6.0, topk: int = 32, warmup: int = 3):
        """
        lam       weight on the transition term relative to log-objectness
        fwd_px    expected forward motion per frame, in unrolled pixels
        sigma_px  tolerance around that motion
        jump_logp floor log-probability for an arbitrary jump; the model can
                  still take one when the evidence is strong enough
        topk      candidates considered per frame
        warmup    frames at the start of a piece decoded by plain argmax, while
                  there is no reliable previous position to condition on
        """
        self.lam = lam
        self.fwd_px = fwd_px
        self.sigma_px = sigma_px
        self.jump_logp = jump_logp
        self.topk = topk
        self.warmup = warmup
        self._state = {}          # piece -> (prev_x, prev_y, n_seen)

    def reset(self, piece: str = None):
        if piece is None:
            self._state = {}
        else:
            self._state.pop(piece, None)

    def _transition_logp(self, x_cand: torch.Tensor, prev_x: float) -> torch.Tensor:
        """Heavy-tailed prior over displacement, in unrolled pixels."""
        d = x_cand - prev_x
        # Gaussian bump centred on the expected forward motion
        near = -0.5 * ((d - self.fwd_px) / self.sigma_px) ** 2
        # floor: any position remains reachable, at a cost
        return torch.maximum(near, torch.full_like(near, self.jump_logp))

    def decode(self, cand_xywh: torch.Tensor, cand_obj: torch.Tensor,
               piece: str, staff_coords=None, add_per_staff=None) -> torch.Tensor:
        """cand_xywh (K,4), cand_obj (K,) -> chosen box (4,).

        `staff_coords`/`add_per_staff` let the transition be computed on the
        UNROLLED x, which is the coordinate the metric actually uses: without
        unrolling, moving from the end of one staff to the start of the next
        looks like a huge backward jump and would be penalised as an error when
        it is in fact the single most ordinary thing the tracker does.
        """
        if cand_obj.numel() == 0:
            return cand_xywh.new_zeros(4)

        k = min(self.topk, cand_obj.shape[0])
        obj, idx = cand_obj.topk(k)
        boxes = cand_xywh[idx]

        prev = self._state.get(piece)
        if prev is None or prev[2] < self.warmup:
            best = int(obj.argmax())
            chosen = boxes[best]
            n = 1 if prev is None else prev[2] + 1
            self._state[piece] = (self._unroll(chosen, staff_coords, add_per_staff),
                                  float(chosen[1]), n)
            return chosen

        prev_x = prev[0]
        x_un = torch.stack([
            torch.as_tensor(self._unroll(b, staff_coords, add_per_staff),
                            device=boxes.device, dtype=boxes.dtype)
            for b in boxes])

        log_obj = torch.log(obj.clamp_min(1e-8))
        total = log_obj + self.lam * self._transition_logp(x_un, prev_x)
        best = int(total.argmax())
        chosen = boxes[best]
        self._state[piece] = (float(x_un[best]), float(chosen[1]), prev[2] + 1)
        return chosen

    @staticmethod
    def _unroll(box, staff_coords, add_per_staff) -> float:
        """(x, y, w, h) -> unrolled x, matching compute_batch_stats exactly."""
        x, y = float(box[0]), float(box[1])
        if staff_coords is None or add_per_staff is None:
            return x
        sc = np.asarray(staff_coords)
        sid = int(np.argmin(np.abs(sc - y)))
        return x + float(np.asarray(add_per_staff)[sid])
