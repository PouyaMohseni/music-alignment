"""What IS the ceiling, once the selector has to behave like a tracker?

The 99.7 figure is a per-frame oracle: it picks each frame's best candidate
independently, so it may leap backwards and forwards through the score. No
causal tracker can do that. Three tighter and more honest ceilings:

  free        per-frame argmin |t - t_gt|                      (the 99.7)
  monotone    the best NON-DECREASING path through the same candidates, chosen
              with full knowledge of the future. Music moves forward, so this
              is the ceiling for any method that respects that -- ours does.
  causal      forward-only greedy that must not move backwards and cannot
              revise: a lower bound on what an online monotone tracker can do.

Ground truth here is monotone to 0.07% (three sub-frame steps in one piece), so
the constraint describes the data rather than being imposed on it.

And the same three at the tight thresholds, because 0.5 s asks "did it find the
right place" while 0.05 s asks "how precisely", and those can have completely
different ceilings.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

FPS = 22050.0 / 1102.0


def load(path):
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        nm, f = k.split('||')
        out.setdefault(nm, {})[f] = z[k]
    for nm, d in out.items():
        lens = d['lens']
        off = np.concatenate([[0], np.cumsum(lens)])
        d['cands'] = [d['t_cand'][off[i]:off[i + 1]] for i in range(len(lens))]
    return out


def monotone_best(cands, t_gt, th):
    """Longest-hitting non-decreasing selection. dp over candidates sorted by
    position, carrying a running max of the previous frame's best score."""
    prev_t = np.array([-np.inf])
    prev_v = np.array([0.0])
    for i, c in enumerate(cands):
        if c.size == 0:
            continue
        o = np.argsort(c)
        t = c[o]
        hit = (np.abs(t - t_gt[i]) <= th).astype(np.float64)
        # best previous value among positions <= t  (prev_t is sorted)
        run = np.maximum.accumulate(prev_v)
        j = np.searchsorted(prev_t, t, side='right') - 1
        # j < 0 means this candidate sits behind EVERY reachable position, so
        # the path cannot pass through it. Marking that -inf keeps the DP exact;
        # the earlier version scored it 0, which silently allowed a backward
        # jump that merely forfeited its accumulated credit.
        base = np.where(j >= 0, run[np.clip(j, 0, len(run) - 1)], -np.inf)
        if not np.isfinite(base).any():
            # 2.1% of frames offer NOTHING at or ahead of every reachable
            # position. A real tracker must still emit a box, so it emits the
            # furthest-forward one and carries on; the DP has to be allowed the
            # same move or it is not measuring the same class of policy. The
            # earlier version restarted the path here and threw away all its
            # accumulated credit, which is what made the offline optimum score
            # BELOW an online greedy -- an impossibility that gave it away.
            base = np.full_like(base, run[-1])
        v = base + hit
        prev_t, prev_v = t, v
    return float(np.max(prev_v[np.isfinite(prev_v)])) if prev_v.size else 0.0


def causal_greedy(cands, t_gt, th):
    """Forward-only: at each frame take the candidate closest to ground truth
    among those not behind the last choice. Never revises."""
    last, hits = -np.inf, 0
    for i, c in enumerate(cands):
        if c.size == 0:
            continue
        ok = c[c >= last - 1e-9]
        # no forward candidate: the tracker must still emit something and must
        # not go back, so it takes the furthest-forward box available
        pick = ok[np.argmin(np.abs(ok - t_gt[i]))] if ok.size else c.max()
        hits += abs(pick - t_gt[i]) <= th
        last = max(last, float(pick))
    return hits


def main(paths):
    pieces = {}
    for p in paths:
        pieces.update(load(p))
    print(f'{len(pieces)} pages\n')
    print(f'{"threshold":>10} {"frames":>8} {"top1":>7} {"free":>7} '
          f'{"monotone":>9} {"causal":>8}')
    for th_s in (0.05, 0.1, 0.5, 1.0):
        th = th_s * FPS
        n = t1 = fr = mo = ca = 0
        for d in pieces.values():
            g = d['t_gt']
            cs = d['cands']
            for i, c in enumerate(cs):
                if c.size == 0:
                    continue
                n += 1
                e = np.abs(c - g[i])
                t1 += e[0] <= th
                fr += e.min() <= th
            mo += monotone_best(cs, g, th)
            ca += causal_greedy(cs, g, th)
        f = lambda v: 100.0 * v / max(n, 1)
        print(f'{th_s:>9.2f}s {n:>8} {f(t1):>7.1f} {f(fr):>7.1f} '
              f'{f(mo):>9.1f} {f(ca):>8.1f}')


if __name__ == '__main__':
    main(sorted(glob.glob(sys.argv[1])))
