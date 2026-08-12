"""D1 -- causal particle-filter score follower over AMT note events.

WHY THIS EXISTS. The decomposed route (audio -> notes, image -> notes, align)
is not the project's goal -- it routes through a symbolic representation, which
the end-to-end target explicitly avoids.  It is built as the comparison point,
because our own measurements say the two halves are nearly solved and the whole
gap is in one place:

    offline DTW over AMT notes, room .............. 98.06 pct@0.5s
    online greedy over the SAME notes, room ....... 10.73  (median err 8-13 s)

An 87-point spread on identical inputs.  The greedy matcher is a pointer with a
24-note window and no way back: once it mis-advances it never recovers, which
is exactly what a median error of eight seconds looks like.  Offline DTW is not
a fix -- it is non-causal, and on this data it is close to tautological because
the recordings are reproducing-piano playback of the score MIDI and contain no
tempo deviation for the alignment to resolve.

So the question this file answers is narrow and worth answering: how much of
that 87 points can a CAUSAL tracker recover?

METHOD.  A particle filter over (score position, tempo):

  state      s = position in merged-onset index units, v = onsets per second
  predict    s <- s + v*dt,  v <- v * lognormal(0, sigma_v)
  update     weight by whether the pitches detected in this frame appear near s
  resample   systematic, when effective sample size drops below half

Three properties the greedy matcher lacked:
  * multi-modality -- particles can track two hypotheses at once, which is what
    a repeated passage requires;
  * recovery -- a `jump` fraction of particles is reseeded from the prior each
    step, so a lost tracker can re-acquire instead of drifting forever;
  * tempo as an explicit state rather than an assumption.

Scored through the SAME metric path as every other experiment
(scripts/amt_bridge_eval.py), so numbers are directly comparable to our 56.6
and to cyolo_sb's 79.9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.amt_bridge_eval import (REC, FPS, THRESHOLDS, load_piece,  # noqa: E402
                                     pct_table)


def _build_pitch_grid(notes, grid_dt=0.05, pad_s=0.25):
    """Piano-roll of the SCORE over its own timeline: (G, 128) bool.

    `notes` is [onset_s, pitch, duration_s, vel, ch].  A pitch is marked
    sounding for its whole duration, plus `pad_s` after onset even for very
    short notes, because an AMT onset detection can land slightly late.

    This replaces the previous observation model, which took ONE pitch per
    merged onset and therefore matched against a fraction of every chord --
    exactly the polyphony problem Nakamura's merged-output HMM formalises.
    """
    dur = float(notes[:, 0].max() + max(notes[:, 2].max(), pad_s)) + 1.0
    G = int(np.ceil(dur / grid_dt)) + 1
    grid = np.zeros((G, 128), dtype=bool)
    for on, p, d, *_ in notes:
        a = int(on / grid_dt)
        b = int((on + max(float(d), pad_s)) / grid_dt) + 1
        grid[max(0, a):min(G, b), int(p) % 128] = True
    return grid, grid_dt


def particle_track(P, det_time, det_pitch, n_particles=1200, sigma_v=0.06,
                   jump_prob=0.015, obs_sharp=6.0, seed=0):
    """Causal particle filter over (score TIME, tempo ratio).

    WHAT CHANGED, AND WHY THE FIRST VERSION SCORED 49.6 WITH PERFECT INPUT
    ---------------------------------------------------------------------
    1. STATE SPACE.  The first version tracked position in *merged-onset index*
       units with tempo in "onsets per second".  Onset spacing in real music is
       wildly uneven, so a constant onset-rate model is badly mis-specified: it
       drifts through every sparse passage.  The literature tracks position in
       SCORE TIME with tempo as a ratio, which is what this does:
           s <- s + v*dt,   v ~ lognormal random walk around 1.0
       s is score seconds, v is dimensionless (performance speed / score speed).
    2. OBSERVATION MODEL.  Was one pitch per merged onset, i.e. a fraction of
       each chord.  Now a full score piano-roll (_build_pitch_grid) giving every
       pitch SOUNDING at score time s, compared two-way against the detected
       pitch set -- a multi-pitch frame model, which is what the particle-filter
       score-following literature uses.
    3. RESTS.  Frames with no detections previously left the weights untouched,
       so the filter free-ran on the tempo prior through every silence.  Silence
       is now evidence: it is consistent with a resting score position and
       inconsistent with a dense one.

    Causal throughout: at frame f only detections with time <= f/FPS are used.
    """
    rng = np.random.default_rng(seed)
    onsets_u = P['onsets_u']                 # (M,) onset FRAME per merged onset
    x_u = P['x_unrolled_u']                  # (M,) unrolled x per merged onset
    notes = P['notes']
    M = len(onsets_u)
    if M < 2 or len(notes) < 2:
        return None

    grid, gdt = _build_pitch_grid(notes)
    G = grid.shape[0]
    n_sounding = grid.sum(1).astype(np.float32)

    # score time of each merged onset, for mapping state -> pixel
    onset_time_u = onsets_u.astype(np.float64) / FPS
    score_end = float(notes[:, 0].max())

    n_frames = int(onsets_u[-1]) + 1
    dt = 1.0 / FPS

    det_frame = np.floor(np.asarray(det_time) * FPS).astype(int)
    by_frame = {}
    for f, p in zip(det_frame, np.asarray(det_pitch, dtype=int)):
        by_frame.setdefault(int(f), []).append(int(p) % 128)

    s = np.abs(rng.normal(0, 0.15, n_particles))          # score seconds
    v = np.exp(rng.normal(0, 0.15, n_particles))          # tempo ratio ~1
    w = np.full(n_particles, 1.0 / n_particles)

    out = np.zeros(n_frames, dtype=np.float64)
    for f in range(n_frames):
        # ---- predict
        s = s + v * dt
        v = np.clip(v * np.exp(rng.normal(0, sigma_v, n_particles)), 0.25, 4.0)
        n_jump = int(jump_prob * n_particles)
        if n_jump:                                        # recovery / repeats
            idx = rng.choice(n_particles, n_jump, replace=False)
            s[idx] = rng.uniform(0, score_end, n_jump)
            v[idx] = np.exp(rng.normal(0, 0.15, n_jump))
        s = np.clip(s, 0.0, score_end)

        # ---- update (vectorised over particles)
        gi = np.clip((s / gdt).astype(int), 0, G - 1)
        obs = by_frame.get(f)
        if obs:
            obs_idx = np.unique(np.asarray(obs, dtype=int))
            # recall: what fraction of detected pitches the score expects here
            hit = grid[gi][:, obs_idx].sum(1) / len(obs_idx)
            # precision: penalise positions expecting far more than we heard,
            # which is what stops the filter parking on a dense chord forever
            expect = np.maximum(n_sounding[gi], 1.0)
            prec = np.minimum(1.0, len(obs_idx) / expect)
            loglik = obs_sharp * (0.75 * hit + 0.25 * prec)
        else:
            # silence: favours positions the score also expects to be quiet
            loglik = -obs_sharp * 0.25 * np.minimum(n_sounding[gi], 4.0) / 4.0

        w = w * np.exp(loglik - loglik.max())
        tot = w.sum()
        w = np.full(n_particles, 1.0 / n_particles) if tot <= 0 else w / tot

        if 1.0 / np.sum(w ** 2) < n_particles / 2:
            pos = (rng.random() + np.arange(n_particles)) / n_particles
            idx = np.clip(np.searchsorted(np.cumsum(w), pos), 0, n_particles - 1)
            s, v = s[idx].copy(), v[idx].copy()
            w = np.full(n_particles, 1.0 / n_particles)

        # ---- output: MAP particle, not the weighted mean.  With two live
        # hypotheses (a repeated passage) the mean lands in the gap between
        # them, which is the failure this filter exists to avoid.
        best = int(np.argmax(w))
        k = int(np.clip(np.searchsorted(onset_time_u, s[best]) - 1, 0, M - 1))
        out[f] = x_u[k]

    return out


def evaluate(P, pred_x_per_frame, latency=0.0):
    """Score at GT onset frames with the CPJKU metric (frame diff / fps)."""
    onsets_u, x_u, c2o = P['onsets_u'], P['x_unrolled_u'], P['c2o']
    diffs = []
    for i, fr in enumerate(onsets_u):
        t = int(fr + round(latency * FPS))
        t = int(np.clip(t, 0, len(pred_x_per_frame) - 1))
        diffs.append(abs(float(c2o(pred_x_per_frame[t])) - float(c2o(x_u[i]))))
    return np.asarray(diffs) / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amt_root', default='/scratch/pmohseni/amt_out')
    ap.add_argument('--model', default='kong_stock')
    ap.add_argument('--tier', default='room')
    ap.add_argument('--out', default='results/analysis/decomposed_tracker.json')
    ap.add_argument('--n_particles', type=int, default=2000)
    ap.add_argument('--jump_prob', type=float, default=0.02)
    ap.add_argument('--oracle', action='store_true',
                    help='feed GT MIDI as detections -- measures the FILTER alone')
    a = ap.parse_args()

    with open(os.path.join(REC, 'rp_split.yaml'), 'rb') as fp:
        pieces = yaml.safe_load(fp)['files']

    print(f'model={a.model} tier={a.tier} oracle={a.oracle} '
          f'particles={a.n_particles} jump={a.jump_prob}', flush=True)

    pooled, per = [], {}
    for p in pieces:
        P = load_piece(p)
        if a.oracle:
            dt_, dp_ = P['notes'][:, 0].astype(float), P['notes'][:, 1].astype(int)
        else:
            f = os.path.join(a.amt_root, a.model, a.tier, p + '.json')
            if not os.path.exists(f):
                print(f'MISSING {f}', flush=True)
                continue
            d = json.load(open(f))
            o = np.asarray(d['onset'], dtype=float)
            q = np.asarray(d['pitch'], dtype=int)
            srt = np.argsort(o, kind='stable')
            dt_, dp_ = o[srt], q[srt]

        pred = particle_track(P, dt_, dp_, n_particles=a.n_particles,
                              jump_prob=a.jump_prob)
        if pred is None:
            continue
        d = evaluate(P, pred)
        pooled.append(d)
        per[p] = dict(pct=pct_table(d), median_err_s=round(float(np.median(d)), 4), n=len(d))
        print(f'  {p[:52]:52s} pct@0.5s={pct_table(d)["0.5"]:6.2f} '
              f'med={np.median(d):.2f}s', flush=True)

    allp = np.concatenate(pooled)
    tbl = pct_table(allp)
    print(f'\nPOOLED pct@0.5s = {tbl["0.5"]:.2f}   median_err={np.median(allp):.3f}s')
    print('  ' + '  '.join(f'<={t}: {tbl[str(t)]}' for t in THRESHOLDS))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({'meta': vars(a), 'pooled': tbl, 'per_piece': per}, open(a.out, 'w'), indent=1)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
