"""What do the pages vel_p8 makes WORSE have in common?

The paired bootstrap put vel_p8 at +1.69 micro over ir_only with a 95% CI of
[-0.95, +3.62]: not resolvable. The point estimate is not evenly earned either
-- 11 pages improve, 8 get worse, and a single Chopin page swings +27.4, which
is most of the headline on its own.

So the question is not "is the mean up" but "are the losses a DIFFERENT failure
from the wins". Two possibilities with opposite consequences:

  SCATTERED flips  -- isolated onsets flipping correct->wrong, i.e. the score
                      is slightly noisier. Unfixable by decoding; you would
                      need a better scorer.
  CONTIGUOUS runs  -- the tracker loses the position and stays lost for a
                      stretch. That is error propagation, it has a beginning
                      you can find, and it is exactly what a recovery mechanism
                      addresses.

Episode structure separates them, and the two arms are paired per onset, so the
comparison is exact rather than distributional.
"""
from __future__ import annotations

import argparse

import numpy as np

FPS, THRESH = 20.0, 0.5


def load(path):
    d = np.load(path)
    out = {}
    for k in d.files:
        piece, _, field = k.partition('||')
        out.setdefault(piece, {})[field] = d[k]
    return out


def runs(mask):
    """Contiguous True episodes as (start, length)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--c2', required=True)
    a = ap.parse_args()
    B, C = load(a.baseline), load(a.c2)
    keys = sorted(set(B) & set(C))

    rows = []
    for k in keys:
        b = B[k]['frame_diff'] / FPS <= THRESH
        c = C[k]['frame_diff'] / FPS <= THRESH
        rows.append((k, b, c, 100.0 * (c.mean() - b.mean())))
    rows.sort(key=lambda r: r[3])

    print('=== episode structure of the FLIPS, worst pages first ===')
    print(f'{"page":34s} {"d_pct":>7s} {"lost":>5s} {"gain":>5s} '
          f'{"ep":>3s} {"longest":>8s} {"first@":>7s}  where')
    tot_lost_runs, tot_lost, agg = [], 0, []
    for k, b, c, d in rows:
        lost, gained = b & ~c, ~b & c
        r = runs(lost)
        longest = max((x[1] for x in r), default=0)
        first = r[0][0] / len(b) if r else float('nan')
        pos = np.mean([s / len(b) for s, _ in r]) if r else float('nan')
        flag = '  <-- REGRESSED' if d < -0.5 else ''
        print(f'{k[:34]:34s} {d:+7.1f} {lost.sum():5d} {gained.sum():5d} '
              f'{len(r):3d} {longest:8d} {first:7.2f}  mean pos {pos:.2f}{flag}')
        if d < -0.5:
            tot_lost_runs += [x[1] for x in r]
            tot_lost += int(lost.sum())
            agg += [s / len(b) for s, _ in r]

    print(f'\n=== the {sum(1 for r in rows if r[3] < -0.5)} regressed pages pooled ===')
    if tot_lost_runs:
        L = np.array(tot_lost_runs)
        print(f'  onsets lost              {tot_lost}')
        print(f'  episodes                 {len(L)}')
        print(f'  median episode length    {np.median(L):.0f} onsets')
        print(f'  longest episode          {L.max()} onsets')
        print(f'  singletons (len 1)       {int((L == 1).sum())} '
              f'({100.0 * (L == 1).mean():.0f}% of episodes, '
              f'{100.0 * L[L == 1].sum() / L.sum():.0f}% of lost onsets)')
        print(f'  onsets in episodes >=5   {int(L[L >= 5].sum())} '
              f'({100.0 * L[L >= 5].sum() / L.sum():.0f}% of lost onsets)')
        print(f'  mean start position      {np.mean(agg):.2f} through the page')
        verdict = ('CONTIGUOUS -- losses are tracking-loss episodes, so they '
                   'have a trigger and a recovery is addressable'
                   if L[L >= 5].sum() > L.sum() * 0.5 else
                   'SCATTERED -- losses are isolated onsets, i.e. a noisier '
                   'score rather than a lost tracker')
        print(f'\n  VERDICT: {verdict}')


if __name__ == '__main__':
    main()
