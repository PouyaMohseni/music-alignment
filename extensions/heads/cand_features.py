"""The ONE implementation of the selector's input features.

Training reads them from a dump; inference computes them inside the decoder.
If those two ever disagree the model silently scores garbage at test time and
every number downstream is wrong, so both call this function. Nothing here may
depend on torch: the decoder runs in numpy.

DESIGN CONSTRAINT: TRAIN ON SYNTH, RUN ON ROOM
----------------------------------------------
The selector is fit on the 353-piece synthetic training split and applied to
real room recordings, so absolute objectness shifts under it. Every feature is
therefore either FRAME-RELATIVE (objectness measured against the best candidate
in the same frame, rank within the frame) or GEOMETRIC (displacement, box size,
containment in the predicted bar and system boxes). Absolute log-objectness is
included but is the only such feature, and `use_abs_obj=False` drops it so the
dependence can be measured rather than assumed.

Nothing here may encode piece identity: no absolute x, no piece length, no
frame index. The unit of data is a (frame, candidate) pair and the model must
not be able to tell which piece it is looking at.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    'log_obj',            # absolute confidence (the one domain-sensitive feature)
    'log_obj_rel',        # confidence relative to the best candidate in this frame
    'rank_frac',          # position in the objectness ordering, in [0, 1)
    'log_rank',
    'log_ncand',          # how crowded this frame is
    'log_w', 'log_h',     # box geometry
    'd_norm',             # displacement / expected travel for this elapsed time
    'd_tanh50',           # raw displacement, two scales, so the net can see both
    'd_tanh200',          # a few-pixel correction and a jump to another system
    'd_is_back',          # moving backwards at all is rare and informative
    'dt_ratio',           # elapsed frames / reference step
    'dy_tanh',            # vertical move (staff change)
    'same_staff',
    'sys_off',            # signed offset from the predicted system box centre
    'sys_in',             # inside that box
    'sys_obj',
    'bar_off',            # same for the predicted bar box, a much tighter region
    'bar_in',
    'bar_obj',
    # SHORT-HORIZON MOTION. Everything above sees exactly one past position, so
    # the ranker has no notion of how fast the music is moving -- only where it
    # was. 60.8% of remaining errors are timing drift within two bars, which is
    # what a velocity estimate is for. These stay per-candidate (each box has
    # its own offset from the extrapolation), which is the property that made
    # backbone features work and z useless.
    #
    # This is NOT the constant-velocity decode that failed before: that blended
    # an extrapolation into the observation on every frame, degrading precision
    # everywhere. Here it is a feature the model may weight or ignore.
    'd_extrap',           # offset from a constant-velocity prediction
    'd_extrap_tanh',
    'v_ratio',            # implied speed relative to the recent speed
    'has_vel',            # whether a velocity estimate exists at all
    # IS THIS A REAL NOTEHEAD, rather than is this where I expect one.
    #
    # Every feature above answers the second question. The first is unasked,
    # and the failure analysis says it is the one that matters: 71% of lost
    # onsets sit in contiguous episodes, and what gives an episode away is that
    # the chosen box has low objectness (AUC 0.805) -- the tracker is pinned
    # somewhere there is no music and must take whatever box is there. Position
    # cannot reveal this (resid AUC 0.558) because each step is still a small
    # plausible move from a wrong place.
    #
    # log_ncand already counts candidates, but it is identical for every
    # candidate in a frame, and a value constant across a frame cannot rank
    # anything. These are its per-candidate versions: a real notehead sits in a
    # cluster of others at sane spacing and is the same size as its neighbours;
    # a spurious box is isolated, or oddly shaped, or far from the bar.
    'nbr_25',             # other candidates within 25 unrolled px
    'nbr_100',
    'gap_l',              # spacing to the nearest candidate either side
    'gap_r',
    'x_rank_frac',        # position in READING order, not objectness order
    'w_rel',              # box size against this frame's median box
    'h_rel',
    'aspect',
    'bar_frac',           # how far through the predicted bar it sits
    # TRACKED TEMPO (Cont 2010's coupled tempo agent, as features).
    #
    # Hard-wiring a tracked v_hat into the prior's mean LOSES: room 93.42 ->
    # 90.62, because step sizes vary 6x within a piece (p10 1.59, p90 9.49
    # px/frame) and centring a Gaussian on the median penalises the many short
    # steps. The conservative fwd_px=6.0 wins by leaving the decision to
    # objectness.
    #
    # That is exactly the shape of the velocity result: constant-velocity
    # DECODE failed, the same quantity as a FEATURE worked, because a feature
    # can be weighted per frame and a prior commits on every one. So the tempo
    # estimate is offered here rather than imposed. It differs from d_extrap in
    # having MEMORY -- an EMA over the piece rather than a two-point estimate,
    # which is what makes it survive the frames either side of a bad step.
    'd_tempo',            # offset from tempo dead reckoning, in expected steps
    'd_tempo_tanh',
    'tempo_rel',          # this candidate's implied speed against tracked tempo
    'has_tempo',          # whether the estimate is established yet
)
NF = len(FEATURE_NAMES)

REF_FRAMES = 5.0          # median onset-to-onset gap, measured on the room set
FWD_PX = 6.0              # shipped prior's expected travel per reference step


MAXK = 256                # must match the dump's cap, or crowding features lie


def build(cand, bar, sys, x_prev, y_prev, dframes, ntot=None, use_abs_obj=True,
          x_prev2=None, dframes_prev=None, v_hat=None):
    """cand: (K, 6) as dumped -- [xu, y, w, h, obj, t]. Returns (K, NF) float32.

    x_prev/y_prev may be None on the first scored frame of a piece, in which
    case every displacement feature is zeroed rather than invented.
    ntot is the candidate count before capping; K may be smaller.
    """
    K = cand.shape[0]
    f = np.zeros((K, NF), np.float32)
    if K == 0:
        return f
    xu, y, w, h, obj = (cand[:, 0], cand[:, 1], cand[:, 2], cand[:, 3], cand[:, 4])

    lo = np.log(np.clip(obj, 1e-8, None))
    f[:, 0] = lo if use_abs_obj else 0.0
    f[:, 1] = lo - lo.max()
    rank = np.arange(K, dtype=np.float32)          # dump is sorted by objectness
    f[:, 2] = rank / max(K, 1)
    f[:, 3] = np.log1p(rank)
    # crowding uses the count BEFORE the cap; K is only how many we score
    f[:, 4] = np.log1p(K if ntot is None else ntot)
    f[:, 5] = np.log(np.clip(w, 1e-3, None))
    f[:, 6] = np.log(np.clip(h, 1e-3, None))

    dt = REF_FRAMES if (dframes is None or dframes <= 0) else float(dframes)
    ratio = dt / REF_FRAMES
    f[:, 11] = np.log1p(ratio)
    if x_prev is not None:
        d = xu - float(x_prev)
        f[:, 7] = d / max(FWD_PX * ratio, 1e-3)
        f[:, 8] = np.tanh(d / 50.0)
        f[:, 9] = np.tanh(d / 200.0)
        f[:, 10] = (d < 0).astype(np.float32)
        if y_prev is not None:
            dy = y - float(y_prev)
            f[:, 12] = np.tanh(dy / 20.0)
            f[:, 13] = (np.abs(dy) < 10.0).astype(np.float32)

    # velocity from the two previous positions, projected forward over the gap
    if (x_prev is not None and x_prev2 is not None and dframes_prev
            and dframes_prev > 0):
        v = (float(x_prev) - float(x_prev2)) / float(dframes_prev)
        pred = float(x_prev) + v * dt
        e = xu - pred
        f[:, 20] = np.clip(e / max(FWD_PX * ratio, 1e-3), -20.0, 20.0)
        f[:, 21] = np.tanh(e / 50.0)
        if abs(v) > 1e-6:
            f[:, 22] = np.clip((xu - float(x_prev)) / (v * dt), -5.0, 5.0)
        f[:, 23] = 1.0

    # --- per-candidate neighbourhood, shape and bar position ---------------
    dx = np.abs(xu[:, None] - xu[None, :])
    f[:, 24] = np.log1p((dx <= 25.0).sum(1) - 1)
    f[:, 25] = np.log1p((dx <= 100.0).sum(1) - 1)
    order = np.argsort(xu)
    xs = xu[order]
    gl = np.full(K, 500.0, np.float32)
    gr = np.full(K, 500.0, np.float32)
    if K > 1:
        step = np.diff(xs).astype(np.float32)
        gl[order[1:]] = step
        gr[order[:-1]] = step
    f[:, 26] = np.tanh(gl / 50.0)
    f[:, 27] = np.tanh(gr / 50.0)
    f[:, 28] = np.argsort(order).astype(np.float32) / max(K - 1, 1)
    wm, hm = float(np.median(w)), float(np.median(h))
    f[:, 29] = np.log(np.clip(w, 1e-3, None) / max(wm, 1e-3))
    f[:, 30] = np.log(np.clip(h, 1e-3, None) / max(hm, 1e-3))
    f[:, 31] = np.log(np.clip(w, 1e-3, None) / np.clip(h, 1e-3, None))
    if float(bar[4]) > 0:
        bw = max(float(bar[2]), 1e-3)
        f[:, 32] = np.clip((xu - (float(bar[0]) - bw / 2.0)) / bw, -2.0, 3.0)

    # tracked tempo: a persistent px/frame estimate, not the two-point speed
    if v_hat is not None and x_prev is not None and v_hat > 0:
        travel = float(v_hat) * dt
        e = xu - (float(x_prev) + travel)
        f[:, 33] = np.clip(e / max(travel, 1e-3), -20.0, 20.0)
        f[:, 34] = np.tanh(e / 50.0)
        f[:, 35] = np.clip((xu - float(x_prev)) / max(travel, 1e-3), -5.0, 5.0)
        f[:, 36] = 1.0

    for base, box in ((14, sys), (17, bar)):
        cx, _cy, bw, _bh, bobj = (float(box[0]), float(box[1]), float(box[2]),
                                  float(box[3]), float(box[4]))
        half = max(bw / 2.0, 1e-3)
        off = (xu - cx) / half
        f[:, base] = np.clip(off, -4.0, 4.0)
        f[:, base + 1] = (np.abs(off) <= 1.0).astype(np.float32)
        f[:, base + 2] = bobj
    return f
