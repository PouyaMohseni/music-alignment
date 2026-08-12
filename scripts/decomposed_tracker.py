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


def particle_track(P, det_time, det_pitch, n_particles=2000, sigma_v=0.08,
                   jump_prob=0.02, obs_sharp=4.0, match_window=3.0, seed=0):
    """Causal filter. Returns (n_eval_frames,) predicted unrolled-x per frame.

    Everything here is causal: at frame t only detections with time <= t are
    used.  No lookahead anywhere, which is the whole point.
    """
    rng = np.random.default_rng(seed)
    onsets_u = P['onsets_u']                 # (M,) onset FRAME per merged onset
    x_u = P['x_unrolled_u']                  # (M,) unrolled x per merged onset
    notes = P['notes']                       # (K,5) [onset_s, pitch, dur, vel, ch]
    M = len(onsets_u)
    if M < 2:
        return None

    # score pitch content per merged-onset index, for the observation model
    score_pitch = notes[:, 1].astype(int)
    note_idx_u = P['note_idx_u']
    # pitches sounding at each merged onset (the note(s) that start there)
    pitches_at = [[] for _ in range(M)]
    for k, ni in enumerate(note_idx_u):
        if 0 <= k < M:
            pitches_at[k].append(int(score_pitch[min(ni, len(score_pitch) - 1)]))

    n_frames = int(onsets_u[-1]) + 1
    dt = 1.0 / FPS

    # bucket detections by frame
    det_frame = np.floor(np.asarray(det_time) * FPS).astype(int)
    by_frame = {}
    for f, p in zip(det_frame, np.asarray(det_pitch, dtype=int)):
        by_frame.setdefault(int(f), []).append(int(p))

    # init: position 0, tempo from the piece's mean onset rate
    dur_s = max(1e-3, onsets_u[-1] / FPS)
    v0 = M / dur_s
    s = rng.uniform(0, 1.0, n_particles)
    v = v0 * np.exp(rng.normal(0, 0.25, n_particles))
    w = np.full(n_particles, 1.0 / n_particles)

    out = np.zeros(n_frames, dtype=np.float64)
    for f in range(n_frames):
        # ---- predict
        s = s + v * dt
        v = v * np.exp(rng.normal(0, sigma_v, n_particles))
        v = np.clip(v, 0.1 * v0, 10.0 * v0)
        # recovery: reseed a small fraction uniformly over the score
        n_jump = int(jump_prob * n_particles)
        if n_jump:
            idx = rng.choice(n_particles, n_jump, replace=False)
            s[idx] = rng.uniform(0, M, n_jump)
            v[idx] = v0 * np.exp(rng.normal(0, 0.25, n_jump))
        s = np.clip(s, 0, M - 1)

        # ---- update
        obs = by_frame.get(f)
        if obs:
            si = np.rint(s).astype(int)
            loglik = np.zeros(n_particles)
            lo = np.maximum(0, si - int(match_window))
            hi = np.minimum(M - 1, si + int(match_window))
            for j in range(n_particles):
                near = set()
                for q in range(lo[j], hi[j] + 1):
                    near.update(pitches_at[q])
                if near:
                    hit = sum(1 for p in obs if p in near) / len(obs)
                else:
                    hit = 0.0
                loglik[j] = obs_sharp * hit
            w = w * np.exp(loglik - loglik.max())
            tot = w.sum()
            w = np.full(n_particles, 1.0 / n_particles) if tot <= 0 else w / tot

            # ---- resample when degenerate
            if 1.0 / np.sum(w ** 2) < n_particles / 2:
                pos = (rng.random() + np.arange(n_particles)) / n_particles
                idx = np.searchsorted(np.cumsum(w), pos)
                idx = np.clip(idx, 0, n_particles - 1)
                s, v = s[idx], v[idx]
                w = np.full(n_particles, 1.0 / n_particles)

        # ---- output: MAP-ish estimate. Weighted mean would land between modes
        # when two hypotheses are alive, which is the failure we are fixing.
        best = int(np.argmax(w))
        out[f] = x_u[int(np.clip(round(s[best]), 0, M - 1))]

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
