"""D1/D2 -- proper online search over cyolo_sb's detections. Zero parameters.

WHY SEARCH, AND WHY THIS IS THE PART OF THE LITERATURE WE HAD SKIPPED
--------------------------------------------------------------------
Everything we built that ADDED PARAMETERS got worse: MERT 56.6, +cross-attention
35.3/19.3, +DINOv2 2.6, a 12.6x train/val overfit, and S2 (a from-scratch
two-tower aligner) at chance. 354 training pieces cannot constrain them.

The one thing that beat the baseline has no parameters at all: C2, which changes
only how a position is read out of an already-trained network, and is worth
+3.46 pct@0.5s (95% CI [+0.97, +5.67], paired over 16 room pieces).

That is not luck. On identical inputs an offline aligner scores 98.06 where
online greedy scores 10.73 -- an 87-point gap that is entirely the decision
rule. The score-following literature has spent twenty years on exactly this
decision rule, and it is the half we had not imported: Matchmaker (ISMIR 2025)
ships OLTWDixon, OLTWArzt and an HMM decoder, and Pairing Real-Time Piano
Transcription with Symbol-level Tracking (arXiv 2505.05078) uses an adapted
OLTW. None of these add capacity.

WHAT C2 GETS WRONG
------------------
C2 keeps exactly ONE hypothesis. It commits to an argmax every frame and can
never revisit it, so a single wrong commit in a repeated passage is unrecoverable
-- the model has no way to represent "it is probably here, but possibly there".
That is the crudest member of this family, and it is already worth +3.46.

TWO DECODERS, ONE INTERFACE
---------------------------
  BeamDecoder     -- K hypotheses over the detector's own candidate boxes.
                     K=1 reduces EXACTLY to C2, so the beam width is a clean
                     controlled test of whether greedy commitment is the cost.
  BandedViterbi   -- full DP over a discretised unrolled-x grid, banded for
                     speed. This is the HMM decoder of the literature, and
                     unlike the beam it can hold mass on positions that no
                     single candidate box happens to cover this frame.
"""
from __future__ import annotations

import numpy as np
import torch

NEG_INF = -1e9


def _unroll(box, staff_coords, add_per_staff) -> float:
    """(x, y, w, h) -> unrolled x, matching compute_batch_stats exactly."""
    x, y = float(box[0]), float(box[1])
    if staff_coords is None or add_per_staff is None:
        return x
    sc = np.asarray(staff_coords)
    sid = int(np.argmin(np.abs(sc - y)))
    return x + float(np.asarray(add_per_staff)[sid])


class _TransitionPrior:
    """Heavy-tailed prior over per-frame displacement in unrolled pixels.

    Identical to C2's, deliberately: the search strategy is the variable under
    test, so the scoring function must be held fixed for the comparison to mean
    anything.
    """

    def __init__(self, fwd_px=6.0, sigma_px=18.0, jump_logp=-6.0):
        self.fwd_px, self.sigma_px, self.jump_logp = fwd_px, sigma_px, jump_logp

    def __call__(self, d):
        near = -0.5 * ((d - self.fwd_px) / self.sigma_px) ** 2
        return np.maximum(near, self.jump_logp)


class BeamDecoder:
    """Online beam search over per-frame detection candidates."""

    def __init__(self, beam: int = 8, lam: float = 1.0, fwd_px: float = 6.0,
                 sigma_px: float = 18.0, jump_logp: float = -6.0,
                 topk: int = 32, warmup: int = 3):
        self.beam, self.lam, self.topk, self.warmup = beam, lam, topk, warmup
        self.prior = _TransitionPrior(fwd_px, sigma_px, jump_logp)
        self._state = {}          # piece -> (scores, xs, boxes, n_seen)

    def reset(self, piece=None):
        if piece is None:
            self._state = {}
        else:
            self._state.pop(piece, None)

    def decode(self, cand_xywh, cand_obj, piece, staff_coords=None, add_per_staff=None):
        if cand_obj.numel() == 0:
            return cand_xywh.new_zeros(4)
        k = min(self.topk, cand_obj.shape[0])
        obj, idx = cand_obj.topk(k)
        boxes = cand_xywh[idx]
        log_obj = torch.log(obj.clamp_min(1e-8)).cpu().numpy()
        xs = np.array([_unroll(b, staff_coords, add_per_staff) for b in boxes])

        st = self._state.get(piece)
        if st is None or st[3] < self.warmup:
            order = np.argsort(-log_obj)[:self.beam]
            n = 1 if st is None else st[3] + 1
            self._state[piece] = (log_obj[order].copy(), xs[order].copy(),
                                  boxes[order].clone(), n)
            return boxes[int(order[0])]

        h_scores, h_xs, _h_boxes, n_seen = st
        # (H, K): every hypothesis extended by every candidate
        d = xs[None, :] - h_xs[:, None]
        total = h_scores[:, None] + log_obj[None, :] + self.lam * self.prior(d)

        flat = total.ravel()
        take = min(self.beam * 4, flat.size)
        cand = np.argpartition(-flat, take - 1)[:take]
        cand = cand[np.argsort(-flat[cand])]
        # Dedupe by candidate box: several hypotheses converging on one detection
        # are the SAME state going forward, and keeping both would collapse an
        # 8-wide beam into one distinct position with 8 slots.
        seen, keep = set(), []
        for f in cand:
            j = int(f % k)
            if j in seen:
                continue
            seen.add(j)
            keep.append((float(flat[f]), j))
            if len(keep) == self.beam:
                break

        new_scores = np.array([s for s, _ in keep])
        sel = np.array([j for _, j in keep])
        new_scores = new_scores - new_scores.max()      # keep bounded
        self._state[piece] = (new_scores, xs[sel], boxes[sel], n_seen + 1)
        return boxes[int(sel[0])]


class BandedViterbi:
    """Online banded Viterbi over a discretised unrolled-x grid (HMM decoder).

    Unlike the beam, the state is a distribution over POSITIONS rather than over
    detected boxes, so probability mass can sit on a location that this frame's
    detector happened not to propose. The band keeps it O(band) per frame.
    """

    def __init__(self, bin_px: float = 8.0, band_px: float = 400.0, lam: float = 1.0,
                 fwd_px: float = 6.0, sigma_px: float = 18.0, jump_logp: float = -6.0,
                 topk: int = 32, warmup: int = 3, floor_logp: float = -12.0):
        self.bin_px, self.band_px = bin_px, band_px
        self.lam, self.topk, self.warmup = lam, topk, warmup
        self.floor_logp = floor_logp
        self.prior = _TransitionPrior(fwd_px, sigma_px, jump_logp)
        self._state = {}          # piece -> (a, x0, n_seen)

    def reset(self, piece=None):
        if piece is None:
            self._state = {}
        else:
            self._state.pop(piece, None)

    def _emission(self, xs, log_obj, grid):
        """Rasterise sparse candidates onto the grid: each cell takes the best
        objectness among candidates falling in it, everything else a floor."""
        e = np.full(grid.size, self.floor_logp)
        b = np.clip(((xs - grid[0]) / self.bin_px).astype(int), 0, grid.size - 1)
        np.maximum.at(e, b, log_obj)
        return e

    def decode(self, cand_xywh, cand_obj, piece, staff_coords=None, add_per_staff=None):
        if cand_obj.numel() == 0:
            return cand_xywh.new_zeros(4)
        k = min(self.topk, cand_obj.shape[0])
        obj, idx = cand_obj.topk(k)
        boxes = cand_xywh[idx]
        log_obj = torch.log(obj.clamp_min(1e-8)).cpu().numpy()
        xs = np.array([_unroll(b, staff_coords, add_per_staff) for b in boxes])

        st = self._state.get(piece)
        centre = float(xs[int(np.argmax(log_obj))]) if st is None else st[1]
        n_bins = int(2 * self.band_px / self.bin_px)
        grid = centre - self.band_px + self.bin_px * np.arange(n_bins)

        e = self._emission(xs, log_obj, grid)
        if st is None or st[2] < self.warmup:
            a = e.copy()
            n = 1 if st is None else st[2] + 1
        else:
            a_prev, prev_c, n = st[0], st[1], st[2]
            # re-centre the previous state onto the new grid
            shift = int(round((prev_c - centre) / self.bin_px))
            a_old = np.full(n_bins, NEG_INF)
            src_lo, src_hi = max(0, -shift), min(n_bins, n_bins - shift)
            if src_hi > src_lo:
                a_old[src_lo + shift:src_hi + shift] = a_prev[src_lo:src_hi]
            steps = np.arange(-4, 33)                     # displacement in bins
            tr = self.lam * self.prior(steps * self.bin_px)
            stack = np.full((steps.size, n_bins), NEG_INF)
            for i, s in enumerate(steps):
                if s >= 0:
                    stack[i, s:] = a_old[:n_bins - s] + tr[i] if s else a_old + tr[i]
                else:
                    stack[i, :s] = a_old[-s:] + tr[i]
            a = e + stack.max(axis=0)
            n = n + 1
        a = a - a.max()
        best_bin = int(np.argmax(a))
        best_x = grid[best_bin]
        self._state[piece] = (a, float(best_x), n)
        # the metric needs a BOX, so return the candidate nearest the chosen
        # position -- the grid carries the belief, the detector carries geometry
        return boxes[int(np.argmin(np.abs(xs - best_x)))]
