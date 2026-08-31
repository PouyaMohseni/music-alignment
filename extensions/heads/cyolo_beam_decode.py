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
    """Heavy-tailed prior over per-STEP displacement in unrolled pixels.

    TIME AWARENESS
    --------------
    The prior was written as if consecutive scored frames were equally spaced.
    Under --only_onsets they are not: load_dataset DROPS every non-onset frame,
    so a "step" is one onset to the next, and measured over the 16 room pieces
    (4007 steps) those gaps run

        p1 = 1 frame (0.05 s)   p50 = 5 (0.25 s)   p99 = 19 (0.95 s)   max = 64 (3.2 s)

    A fixed fwd_px=6 / sigma_px=18 therefore expects the same 6 px of travel and
    allows the same 18 px of slack whether 50 ms or 3.2 s of music has gone by.
    For a piece moving at a roughly steady tempo the mean displacement is
    PROPORTIONAL to elapsed time, so the prior actively fights the largest real
    jumps -- exactly the steps where a tracker is most likely to be lost.

    mu_pow / sig_pow set how mean and width scale with the step ratio
    s = dframes / ref_frames:

        mu  = fwd_px   * s**mu_pow
        sig = sigma_px * s**sig_pow

    mu_pow=1 is constant tempo. sig_pow=0.5 is a random walk in tempo, sig_pow=1
    is multiplicative tempo noise. BOTH ZERO REPRODUCES THE SHIPPED PRIOR
    EXACTLY, so the control arm is bit-identical rather than merely close.
    """

    def __init__(self, fwd_px=6.0, sigma_px=18.0, jump_logp=-6.0,
                 mu_pow=0.0, sig_pow=0.0, ref_frames=5.0, clip=(0.2, 8.0)):
        self.fwd_px, self.sigma_px, self.jump_logp = fwd_px, sigma_px, jump_logp
        self.mu_pow, self.sig_pow = mu_pow, sig_pow
        self.ref_frames, self.clip = ref_frames, clip

    @property
    def time_aware(self):
        return self.mu_pow != 0.0 or self.sig_pow != 0.0

    def scale_of(self, dframes):
        if dframes is None or not self.time_aware:
            return 1.0
        return float(np.clip(dframes / self.ref_frames, *self.clip))

    def __call__(self, d, s=1.0):
        mu = self.fwd_px * s ** self.mu_pow
        sg = self.sigma_px * s ** self.sig_pow
        near = -0.5 * ((d - mu) / sg) ** 2
        return np.maximum(near, self.jump_logp)


class BeamDecoder:
    """Online beam search over per-frame detection candidates."""

    def __init__(self, beam: int = 8, lam: float = 1.0, fwd_px: float = 6.0,
                 sigma_px: float = 18.0, jump_logp: float = -6.0,
                 topk: int = 32, warmup: int = 3, discount: float = 1.0,
                 cluster_px: float = 0.0, mu_pow: float = 0.0,
                 sig_pow: float = 0.0, ref_frames: float = 5.0,
                 reanchor_k: int = 0, reanchor_px: float = 200.0):
        """
        cluster_px  radius for EVIDENCE POOLING (0 = off, decoder unchanged).
        discount  fading memory on the accumulated path score, in [0, 1].

        This is not a tuning knob, it is the difference between a working
        online tracker and a frozen one. Textbook beam search accumulates path
        log-likelihood without bound, which is correct when you decode a whole
        sequence offline and read the best path at the end. Here we must EMIT A
        POSITION EVERY FRAME, and the accumulated spread grows without limit:
        after a few hundred frames a leading hypothesis is ahead by hundreds of
        nats while one frame of fresh evidence is worth a few, so the beam can
        never revise itself. Wider beams then decode a stale commitment with
        more conviction -- which is exactly the monotonic degradation measured
        (84.7 / 84.3 / 84.3 / 83.8 / 83.8 for beam 1/4/8/16/32).

        discount=0 discards history entirely and reduces to C2's greedy rule;
        discount=1 is textbook beam search. Anything between gives the past a
        finite half-life, so evidence can still overturn a wrong commit.
        """
        self.beam, self.lam, self.topk, self.warmup = beam, lam, topk, warmup
        self.discount, self.cluster_px = discount, cluster_px
        self.prior = _TransitionPrior(fwd_px, sigma_px, jump_logp,
                                      mu_pow=mu_pow, sig_pow=sig_pow,
                                      ref_frames=ref_frames)
        self.reanchor_k, self.reanchor_px = reanchor_k, reanchor_px
        self._state = {}          # piece -> (scores, xs, boxes, n_seen)
        self._last_frame = {}     # piece -> spectrogram frame of the last step
        self._stall = {}          # piece -> consecutive steps the evidence disagreed

    def reset(self, piece=None):
        if piece is None:
            self._state = {}
            self._last_frame = {}
            self._stall = {}
        else:
            self._state.pop(piece, None)
            self._last_frame.pop(piece, None)
            self._stall.pop(piece, None)

    def _step_scale(self, piece, frame):
        """Ratio of this step's elapsed time to the reference step.

        Returns 1.0 whenever the frame index is unavailable or the prior is not
        time-aware, so an unpatched dataloader degrades to the shipped
        behaviour instead of silently inventing a scale."""
        if frame is None or not self.prior.time_aware:
            return 1.0
        prev = self._last_frame.get(piece)
        self._last_frame[piece] = frame
        if prev is None or frame <= prev:
            return 1.0
        return self.prior.scale_of(frame - prev)

    def _pool(self, boxes, obj, xs):
        """EVIDENCE POOLING -- treat co-located detections as one hypothesis.

        The detector fires several anchors on the same notehead. A bare argmax
        over objectness asks "which single anchor is most confident?", when the
        question we actually want answered is "which POSITION has the most
        evidence behind it". A spot backed by four medium detections is stronger
        evidence than one backed by a single slightly-higher spike, and the
        current rule cannot express that difference at all.

        So bin candidates by unrolled x, sum their probability mass per bin, and
        represent each bin by its objectness-weighted centroid. Summing in
        PROBABILITY space rather than log space is the point -- it is the total
        mass at that location. The centroid also averages several continuous box
        centres for one note, which cuts the variance of any single anchor's
        regression.

        cluster_px = 0 disables this and the decoder is bit-identical to before.
        """
        if self.cluster_px <= 0 or len(xs) < 2:
            return boxes, obj, xs
        order = np.argsort(xs)
        gb, go, gx = [], [], []
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] - xs[order[i]] <= self.cluster_px:
                j += 1
            grp = order[i:j + 1]
            w = np.clip(obj[grp], 1e-8, None).astype(np.float64)
            gb.append(np.average(boxes[grp], axis=0, weights=w))
            go.append(min(float(w.sum()), 1.0 - 1e-6))   # pooled mass stays a probability
            gx.append(float(np.average(xs[grp], weights=w)))
            i = j + 1
        return np.stack(gb), np.array(go), np.array(gx)

    def decode(self, cand_xywh, cand_obj, piece, staff_coords=None, add_per_staff=None,
               frame=None, bar=None, sys=None, ntot=None):
        # bar/sys/ntot are accepted and ignored: the caller passes the same
        # kwargs to every decoder so they stay interchangeable, and only the
        # learned scorer reads them.
        if cand_obj.numel() == 0:
            return cand_xywh.new_zeros(4)
        s = self._step_scale(piece, frame)
        k = min(self.topk, cand_obj.shape[0])
        obj, idx = cand_obj.topk(k)
        boxes = cand_xywh[idx]
        log_obj = torch.log(obj.clamp_min(1e-8)).cpu().numpy()
        xs = np.array([_unroll(b, staff_coords, add_per_staff) for b in boxes])

        if self.cluster_px > 0:
            bnp, onp, xs = self._pool(boxes.cpu().numpy(), obj.cpu().numpy(), xs)
            boxes = torch.as_tensor(bnp, dtype=cand_xywh.dtype, device=cand_xywh.device)
            log_obj = np.log(np.clip(onp, 1e-8, None))

        st = self._state.get(piece)
        if st is None or st[3] < self.warmup:
            order = np.argsort(-log_obj)[:self.beam]
            n = 1 if st is None else st[3] + 1
            self._state[piece] = (log_obj[order].copy(), xs[order].copy(),
                                  boxes[order].clone(), n)
            return boxes[int(order[0])]

        h_scores, h_xs, _h_boxes, n_seen = st

        # RECOVERY. 70.9% of our remaining error frames sit in runs of five or
        # more consecutive onsets, some lasting 16 s -- the tracker loses the
        # place and the temporal prior, which is what removed the jitter, is
        # also what holds it there. Compared with the raw baseline we cut the
        # number of error runs from 289 to 126 and made the survivors LONGER.
        #
        # So watch for the evidence disagreeing persistently: if the detector's
        # own most confident box has been far from the tracked position for
        # `reanchor_k` onsets in a row, stop arguing with it and jump. One
        # frame of disagreement is noise; five in a row is being lost.
        if self.reanchor_k > 0:
            top = int(np.argmax(log_obj))
            far = abs(float(xs[top]) - float(h_xs[0])) > self.reanchor_px
            n_stall = self._stall.get(piece, 0) + 1 if far else 0
            self._stall[piece] = n_stall
            if n_stall >= self.reanchor_k:
                self._stall[piece] = 0
                self._state[piece] = (np.zeros(1), xs[top:top + 1],
                                      boxes[top:top + 1], n_seen + 1)
                return boxes[top]

        # (H, K): every hypothesis extended by every candidate
        d = xs[None, :] - h_xs[:, None]
        total = (self.discount * h_scores[:, None]
                 + log_obj[None, :] + self.lam * self.prior(d, s))

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
                 topk: int = 32, warmup: int = 3, floor_logp: float = -12.0,
                 discount: float = 1.0, mu_pow: float = 0.0,
                 sig_pow: float = 0.0, ref_frames: float = 5.0):
        # same unbounded-accumulation problem as the beam: `a` sums evidence
        # over the whole piece, so the belief hardens until fresh frames cannot
        # move it. Discounting gives the past a finite half-life.
        self.bin_px, self.band_px = bin_px, band_px
        self.lam, self.topk, self.warmup = lam, topk, warmup
        self.floor_logp = floor_logp
        self.discount = discount
        self.prior = _TransitionPrior(fwd_px, sigma_px, jump_logp,
                                      mu_pow=mu_pow, sig_pow=sig_pow,
                                      ref_frames=ref_frames)
        self._state = {}          # piece -> (a, x0, n_seen)
        self._last_frame = {}

    def reset(self, piece=None):
        if piece is None:
            self._state = {}
            self._last_frame = {}
        else:
            self._state.pop(piece, None)
            self._last_frame.pop(piece, None)

    _step_scale = BeamDecoder._step_scale

    def _emission(self, xs, log_obj, grid):
        """Rasterise sparse candidates onto the grid: each cell takes the best
        objectness among candidates falling in it, everything else a floor."""
        e = np.full(grid.size, self.floor_logp)
        b = np.clip(((xs - grid[0]) / self.bin_px).astype(int), 0, grid.size - 1)
        np.maximum.at(e, b, log_obj)
        return e

    def decode(self, cand_xywh, cand_obj, piece, staff_coords=None, add_per_staff=None,
               frame=None, bar=None, sys=None, ntot=None):
        if cand_obj.numel() == 0:
            return cand_xywh.new_zeros(4)
        s = self._step_scale(piece, frame)
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
            a_prev, prev_c, n = st[0] * self.discount, st[1], st[2]
            # re-centre the previous state onto the new grid
            shift = int(round((prev_c - centre) / self.bin_px))
            a_old = np.full(n_bins, NEG_INF)
            src_lo, src_hi = max(0, -shift), min(n_bins, n_bins - shift)
            if src_hi > src_lo:
                a_old[src_lo + shift:src_hi + shift] = a_prev[src_lo:src_hi]
            steps = np.arange(-4, 33)                     # displacement in bins
            tr = self.lam * self.prior(steps * self.bin_px, s)
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


class ScorerDecoder:
    """Greedy decode where the scoring function is LEARNED, not hand-tuned.

    The shipped rule is `log p_obj + lam * heavy_tail(displacement)`: four
    constants, all of them chosen by sweeping on the room test set. This
    replaces that expression with a 9.7k-parameter listwise network fit on the
    353-piece TRAINING split, which removes both the hand-tuning and the
    test-set dependence in one step.

    Greedy on purpose. The beam-width sweep degraded monotonically (84.7 / 84.3
    / 84.3 / 83.8 / 83.8 for width 1/4/8/16/32), so beam=1 is the configuration
    that actually wins and there is nothing to gain from carrying the extra
    hypotheses through a learned score as well.

    `blend` mixes the learned score with the shipped prior:
        blend=1  -> learned only
        blend=0  -> the shipped decoder exactly (the control)
    so the comparison has a bit-identical control arm rather than a nearby one.
    """

    def __init__(self, scorer_path: str, topk: int = 256, blend: float = 1.0,
                 lam: float = 1.0, fwd_px: float = 6.0, sigma_px: float = 18.0,
                 jump_logp: float = -6.0, mu_pow: float = 1.0,
                 sig_pow: float = 0.0, ref_frames: float = 5.0):
        import torch as _t

        from extensions.heads.cand_scorer import load
        self.model, self.meta = load(scorer_path)
        self._t = _t
        self.topk, self.blend = topk, blend
        self.lam = lam
        self.prior = _TransitionPrior(fwd_px, sigma_px, jump_logp,
                                      mu_pow=mu_pow, sig_pow=sig_pow,
                                      ref_frames=ref_frames)
        self._state = {}          # piece -> (x_prev, y_prev)
        self._last_frame = {}
        print(f'[SCORER] {scorer_path}: {self.model.n_params} params, '
              f'blend={blend}, topk={topk}, meta={self.meta}', flush=True)

    def reset(self, piece=None):
        if piece is None:
            self._state, self._last_frame = {}, {}
        else:
            self._state.pop(piece, None)
            self._last_frame.pop(piece, None)

    def decode(self, cand_xywh, cand_obj, piece, staff_coords=None,
               add_per_staff=None, frame=None, bar=None, sys=None, ntot=None,
               z=None):
        import numpy as _np

        from extensions.heads.cand_features import build
        if cand_obj.numel() == 0:
            return cand_xywh.new_zeros(4)
        total = int(cand_obj.shape[0]) if ntot is None else int(ntot)
        k = min(self.topk, cand_obj.shape[0])
        obj, idx = cand_obj.topk(k)
        boxes = cand_xywh[idx]
        bnp = boxes.detach().cpu().numpy()
        xs = _np.array([_unroll(b, staff_coords, add_per_staff) for b in bnp])

        prev = self._state.get(piece)
        x_prev, y_prev, x_prev2, f_prev, f_prev2 = (
            (None, None, None, None, None) if prev is None else prev)
        dfr = None if (frame is None or f_prev is None or frame <= f_prev) else frame - f_prev
        dfr_prev = (f_prev - f_prev2
                    if f_prev is not None and f_prev2 is not None and f_prev > f_prev2
                    else None)

        # cand layout must match the dump exactly: [xu, y, w, h, obj, t].
        # `t` (the onset coordinate) is a LABEL, never an input, so it is left
        # zero here -- build() does not read column 5.
        c = _np.zeros((k, 6), _np.float32)
        c[:, 0], c[:, 1] = xs, bnp[:, 1]
        c[:, 2], c[:, 3] = bnp[:, 2], bnp[:, 3]
        c[:, 4] = obj.detach().cpu().numpy()
        bar_u = _np.zeros(5, _np.float32) if bar is None else _np.asarray(bar, _np.float32)
        sys_u = _np.zeros(5, _np.float32) if sys is None else _np.asarray(sys, _np.float32)
        for box in (bar_u, sys_u):
            if box[4] > 0:
                box[0] = _unroll(box, staff_coords, add_per_staff)

        f = build(c, bar_u, sys_u, x_prev, y_prev, dfr, ntot=total,
                  use_abs_obj=self.model.use_abs_obj,
                  x_prev2=x_prev2, dframes_prev=dfr_prev)
        # Checkpoints fitted before a feature was appended expect the old width.
        # FEATURE_NAMES only ever grows at the end, so truncating is exact for
        # them and a no-op for new ones -- without this every selector trained
        # so far would silently mis-shape at inference.
        if f.shape[1] != self.model.nf:
            if f.shape[1] < self.model.nf:
                raise RuntimeError(f'model wants {self.model.nf} features, '
                                   f'build() gives {f.shape[1]}')
            f = f[:, :self.model.nf]
        zz = None
        if self.model.zenc is not None and z is not None:
            zz = self._t.from_numpy(_np.asarray(z, _np.float32)).unsqueeze(0)
        with self._t.no_grad():
            s = self.model(self._t.from_numpy(f).unsqueeze(0), z=zz)[0].numpy()

        if self.blend < 1.0:
            lo = _np.log(_np.clip(c[:, 4], 1e-8, None))
            if x_prev is None:
                hand = lo
            else:
                sc = self.prior.scale_of(dfr)
                hand = lo + self.lam * self.prior(xs - x_prev, sc)
            s = self.blend * s + (1.0 - self.blend) * hand

        j = int(_np.argmax(s))
        self._state[piece] = (float(xs[j]), float(bnp[j, 1]), x_prev, frame, f_prev)
        return boxes[j]
