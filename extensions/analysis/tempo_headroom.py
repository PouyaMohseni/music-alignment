"""Is one global px-per-frame constant defensible, or is tempo per-piece?

The shipped prior expects fwd_px=6.0 pixels of travel per REF_FRAMES=5 frames
-- 1.2 px/frame -- for every piece in the corpus, and the learned scorer sees
the same constant through d_norm. Antescofo (Cont 2010) instead couples a
TEMPO agent to the position agent, so the expected duration of the next event
is a tracked quantity rather than a constant.

Before building that, measure whether the constant is actually wrong. Ground
truth gives the true position at every onset, so the true px/frame is directly
observable. Two questions:

  BETWEEN pieces  if they all sit near 1.2 px/frame, a per-piece tempo buys
                  nothing and the constant is fine.
  WITHIN a piece  if tempo drifts a lot inside one performance, even a
                  per-piece constant is not enough and it has to be tracked.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.offline_decode import load

FWD_PX, REF = 6.0, 5.0


def main():
    pages = load('/scratch/pmohseni/omr/candf/room.npz')
    print(f'assumed constant: {FWD_PX / REF:.2f} px/frame '
          f'({FWD_PX} px per {REF:.0f}-frame step)\n')
    print(f'{"piece":46s} {"n":>5s} {"px/frame":>9s} {"vs 1.2":>7s} '
          f'{"IQR/med":>8s}')
    allv, rows = [], []
    for p in pages:
        xs, fr = [], []
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            j = int(np.argmin(np.abs(c[:, 5] - p['t_gt'][i])))   # oracle box
            xs.append(float(c[j, 0])); fr.append(int(p['frame'][i]))
        xs, fr = np.array(xs), np.array(fr)
        d, dt = np.diff(xs), np.diff(fr).astype(float)
        ok = (dt > 0) & (d > -50)          # ignore page turns and backward jumps
        if ok.sum() < 20:
            continue
        v = d[ok] / dt[ok]
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        rows.append((p['name'][:46], int(ok.sum()), med, med / (FWD_PX / REF),
                     iqr / max(abs(med), 1e-6)))
        allv.append(v)
    rows.sort(key=lambda r: r[2])
    for r in rows:
        print(f'{r[0]:46s} {r[1]:5d} {r[2]:9.2f} {r[3]:7.2f}x {r[4]:8.2f}')
    meds = np.array([r[2] for r in rows])
    print(f'\nBETWEEN pieces: median {np.median(meds):.2f} px/frame, '
          f'range {meds.min():.2f}-{meds.max():.2f}, '
          f'spread {meds.max() / max(meds.min(), 1e-6):.1f}x')
    print(f'  pieces more than 2x off the assumed 1.2: '
          f'{int(((meds / 1.2 > 2) | (meds / 1.2 < 0.5)).sum())} of {len(meds)}')
    w = np.concatenate(allv)
    print(f'WITHIN pieces: median IQR/median = '
          f'{np.median([r[4] for r in rows]):.2f}')
    print(f'  pooled px/frame p10 {np.percentile(w, 10):.2f}  '
          f'p50 {np.percentile(w, 50):.2f}  p90 {np.percentile(w, 90):.2f}')


if __name__ == '__main__':
    main()
