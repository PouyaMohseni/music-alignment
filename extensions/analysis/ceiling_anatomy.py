"""What sets the ceiling, and which lever could move it?

We are at 93.4 against a remembered 96.0 causal ceiling and 99.7 free oracle.
Every mechanism left approaches that ceiling; raising it is a different problem,
and the first question is what binds it. Three candidates, and they imply
completely different work:

  MISSING CANDIDATES   the right box is not in the list at all. Then no decoder
                       helps and the lever is the detector, or a deeper topk.
  MONOTONICITY         the right box is present but only reachable by moving
                       backwards, which an online tracker will not do.
  RANKING              the box is present and reachable, and we simply fail to
                       choose it. That is the decoder's problem and the ceiling
                       is already where it should be.

Measured here rather than assumed:
  * the free oracle as a function of topk -- does a deeper list contain more
    right answers, or has it saturated?
  * the fraction of frames with NO candidate inside the threshold, which is the
    irreducible floor for any decoder whatsoever
  * the offline MONOTONE optimum by dynamic programming, which is the best a
    non-decreasing tracker could do having seen the whole piece
  * the same DP with a backward allowance, which prices monotonicity directly

The DP is O(K log K) per frame: sort the previous frame's candidates by x, take
a prefix maximum, and binary search. The naive O(K^2) over 256 candidates and
4149 frames would be 272M transitions.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.offline_decode import TH, load


def monotone_dp(page, slack=0.0, topk=256):
    """Best achievable by a non-decreasing path over the candidates, offline."""
    prev_x = prev_best = None
    total = 0
    for i, c in enumerate(page['cand']):
        if c.shape[0] == 0:
            continue
        cs = c[:topk]
        hit = (np.abs(cs[:, 5] - page['t_gt'][i]) <= TH).astype(np.int32)
        x = cs[:, 0].astype(np.float64)
        if prev_x is None:
            best = hit.astype(np.int32)
        else:
            o = np.argsort(prev_x)
            xs, vs = prev_x[o], prev_best[o]
            pref = np.maximum.accumulate(vs)          # best over all x' <= x
            k = np.searchsorted(xs, x + slack, side='right') - 1
            base = np.where(k >= 0, pref[np.clip(k, 0, len(pref) - 1)], 0)
            best = base.astype(np.int32) + hit
        prev_x, prev_best = x, best
        total = int(best.max())
    return total


def causal_oracle(page, window=None, topk=256):
    """REAL-TIME ceiling: commit frame by frame, never revise, no future.

    An oracle that knows which candidate is correct NOW but must choose one and
    live with it. Two constraints make this less than 100%:

      monotone   it may not go behind what it already committed, so an early
                 correct choice can strand it when the next correct box is
                 behind (ground truth does step backwards on 0.73% of onsets)
      window     it may not teleport arbitrarily far ahead. An online tracker
                 that has not heard the music yet cannot know to jump there, so
                 an unbounded forward reach is not a real-time ability.

    The offline DP is the counterpart: same candidates, but free to plan the
    whole path, sacrificing one frame to be positioned for a later one.
    """
    x_prev, hit, n = None, 0, 0
    for i, c in enumerate(page['cand']):
        if c.shape[0] == 0:
            continue
        cs = c[:topk]
        n += 1
        ok = np.ones(cs.shape[0], bool)
        if x_prev is not None:
            ok &= cs[:, 0] >= x_prev
            if window is not None:
                ok &= cs[:, 0] <= x_prev + window
        if not ok.any():
            ok = cs[:, 0] >= x_prev if x_prev is not None else np.ones(cs.shape[0], bool)
            if not ok.any():
                continue
        idx = np.flatnonzero(ok)
        err = np.abs(cs[idx, 5] - page['t_gt'][i])
        j = idx[int(np.argmin(err))]
        hit += err.min() <= TH
        x_prev = float(cs[j, 0])
    return hit, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/scratch/pmohseni/omr/candf/room.npz')
    a = ap.parse_args()
    pages = load(a.dump)
    n = sum(1 for p in pages for c in p['cand'] if c.shape[0])
    print(f'{len(pages)} pages, {n} scored onsets\n')

    print('=== is the answer even in the list? (free oracle vs depth) ===')
    print(f'{"topk":>6s} {"oracle":>8s}')
    for k in (1, 4, 16, 32, 64, 128, 256):
        h = sum(int((np.abs(c[:k, 5] - p['t_gt'][i]) <= TH).any())
                for p in pages for i, c in enumerate(p['cand']) if c.shape[0])
        print(f'{k:6d} {100.0 * h / n:8.2f}')
    miss = sum(1 for p in pages for i, c in enumerate(p['cand'])
               if c.shape[0] and not (np.abs(c[:, 5] - p['t_gt'][i]) <= TH).any())
    print(f'\nframes with NO candidate inside the threshold: {miss} '
          f'({100.0 * miss / n:.2f}%) -- the irreducible floor\n')

    print('=== what does monotonicity cost? (offline DP over candidates) ===')
    print(f'{"backward allowance":>20s} {"ceiling":>8s}')
    for slack, lab in ((0.0, 'strict (0 px)'), (25.0, '25 px'), (100.0, '100 px'),
                       (400.0, '400 px'), (1e9, 'unconstrained')):
        tot = sum(monotone_dp(p, slack=slack) for p in pages)
        print(f'{lab:>20s} {100.0 * tot / n:8.2f}')

    print('\n=== REAL-TIME vs NOT REAL-TIME ===')
    print('  real-time: commit each frame, never revise, no future.')
    print('  offline:   same candidates, free to plan the whole path.\n')
    print(f'{"ceiling":>34s} {"value":>8s}')
    for w, lab in ((None, 'real-time, unbounded reach'),
                   (400.0, 'real-time, <=400 px forward'),
                   (200.0, 'real-time, <=200 px forward'),
                   (100.0, 'real-time, <=100 px forward'),
                   (50.0, 'real-time, <=50 px forward')):
        h = nn = 0
        for p_ in pages:
            a_, b_ = causal_oracle(p_, window=w)
            h += a_; nn += b_
        print(f'{lab:>34s} {100.0 * h / max(nn, 1):8.2f}')
    off = sum(monotone_dp(p_, slack=0.0) for p_ in pages)
    fre = sum(monotone_dp(p_, slack=1e9) for p_ in pages)
    print(f'{"offline, monotone (plans ahead)":>34s} {100.0 * off / n:8.2f}')
    print(f'{"offline, unconstrained":>34s} {100.0 * fre / n:8.2f}')
    print(f'\nshipped decoder is at 93.42 on room.')


if __name__ == '__main__':
    main()
