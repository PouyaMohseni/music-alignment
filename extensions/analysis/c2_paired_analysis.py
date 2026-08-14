"""Is C2's +3.5 real, and what is the bar-accuracy drop actually made of?

TWO QUESTIONS, ONE PAIRED SAMPLE
--------------------------------
1. PAIRED BOOTSTRAP. The headline 79.9 -> 83.4 pools every frame from all 16
   pieces into one ratio (MICRO). With ~16 pieces and a large design effect,
   comparing two such numbers as if they were independent is not informative.
   Resampling PIECES with replacement and recomputing BOTH arms on the same
   resample cancels piece difficulty, which is the dominant variance component.

2. WHERE THE BAR ERRORS LAND. Bar accuracy falls 0.829 -> 0.689. Two stories fit
   that, and they have opposite implications:

     (a) the prior drags predictions off-target -- then bar misses should sit on
         frames that are ALSO badly timed, and the tracking gain is suspect;
     (b) the box is temporally right but in a visually identical repeated
         passage -- then bar misses sit on WELL-timed frames, which is the
         repeat ambiguity we already diagnosed, and the gain is real.

   Conditioning bar correctness on frame_diff separates them directly.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

FPS = 20.0
THRESH = 0.5


def load(path):
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        fname, field = k.split('||')
        out.setdefault(fname, {})[field] = z[k]
    return out


def micro(pieces, keys, th=THRESH):
    """Pooled over frames -- the harness's own headline aggregation."""
    d = np.concatenate([pieces[k]['frame_diff'] for k in keys]) / FPS
    return 100.0 * np.mean(d <= th)


def macro(pieces, keys, th=THRESH):
    """Mean of per-piece ratios -- every piece counts once."""
    return 100.0 * np.mean([np.mean(pieces[k]['frame_diff'] / FPS <= th) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--c2', required=True)
    ap.add_argument('--n_boot', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    B, C = load(a.baseline), load(a.c2)
    keys = sorted(set(B) & set(C))
    print(f'pieces present in both arms: {len(keys)}  '
          f'(baseline {len(B)}, c2 {len(C)})')
    if len(keys) != len(B) or len(keys) != len(C):
        print('  WARNING: arms disagree on the piece set; using the intersection')

    # the two arms must have identical frame counts per piece, or they are not
    # paired and nothing below is valid
    bad = [k for k in keys if len(B[k]['frame_diff']) != len(C[k]['frame_diff'])]
    if bad:
        print(f'  FATAL: {len(bad)} pieces have different frame counts between '
              f'arms, e.g. {bad[:3]} -- not a paired sample')
        sys.exit(2)

    print('\n=== point estimates ===')
    for name, fn in (('MICRO (harness headline)', micro), ('MACRO (per-piece mean)', macro)):
        b, c = fn(B, keys), fn(C, keys)
        print(f'  {name:26s} baseline={b:5.1f}  C2={c:5.1f}  delta={c-b:+5.1f}')

    # ---- per-piece deltas: is the gain broad or carried by a few pieces?
    print('\n=== per-piece pct@0.5s ===')
    rows = []
    for k in keys:
        b = 100.0 * np.mean(B[k]['frame_diff'] / FPS <= THRESH)
        c = 100.0 * np.mean(C[k]['frame_diff'] / FPS <= THRESH)
        nb = 100.0 * np.mean(B[k].get('cls_1', np.zeros(1)))
        nc = 100.0 * np.mean(C[k].get('cls_1', np.zeros(1)))
        rows.append((k, len(B[k]['frame_diff']), b, c, c - b, nb, nc, nc - nb))
    rows.sort(key=lambda r: r[4])
    print(f'  {"piece":<34s} {"n":>6s} {"base":>6s} {"C2":>6s} {"d_pct":>7s} '
          f'{"barB":>6s} {"barC2":>6s} {"d_bar":>7s}')
    for k, n, b, c, d, nb, nc, dn in rows:
        print(f'  {k[:34]:<34s} {n:6d} {b:6.1f} {c:6.1f} {d:+7.1f} '
              f'{nb:6.1f} {nc:6.1f} {dn:+7.1f}')
    d_pct = np.array([r[4] for r in rows])
    print(f'  pieces improved: {int((d_pct > 0).sum())}/{len(rows)}   '
          f'unchanged: {int((d_pct == 0).sum())}   hurt: {int((d_pct < 0).sum())}')

    # ---- paired bootstrap, CLUSTERED BY PIECE
    # The recorder keys on file name, and a piece spans several score PAGES
    # (ChopinFF__O9__nocturne appears 5 times, BachJS__BWV830 3 times) -- 16
    # pieces render as 25 pages. Pages of one performance share a recording, a
    # room, a player and a tempo, so resampling PAGES treats correlated units as
    # independent and reports a CI that is too narrow. Resample whole pieces.
    import re
    clusters = {}
    for k in keys:
        clusters.setdefault(re.sub(r'_page_\d+$', '', k), []).append(k)
    cnames = sorted(clusters)
    print(f'\nclustering: {len(keys)} pages -> {len(cnames)} pieces '
          f'(bootstrap resamples PIECES)')

    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(cnames))
    dmi, dma = [], []
    for _ in range(a.n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        ks = [p for i in s for p in clusters[cnames[i]]]
        dmi.append(micro(C, ks) - micro(B, ks))
        dma.append(macro(C, ks) - macro(B, ks))
    for name, arr in (('MICRO', np.array(dmi)), ('MACRO', np.array(dma))):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
        print(f'\n=== paired bootstrap, {name} ({a.n_boot} resamples of pieces) ===')
        print(f'  delta = {arr.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  '
              f'two-sided p = {p:.4f}')
        print(f'  {"CI excludes 0 -- gain is resolvable" if lo > 0 or hi < 0 else "CI includes 0 -- NOT resolvable at this sample size"}')

    # ---- where the bar errors land, conditioned on timing
    print('\n=== bar correctness conditioned on frame timing ===')
    print('  If C2 bar-misses concentrate on WELL-timed frames, the box is')
    print('  temporally right and bar-wrong => repeat ambiguity, gain is real.')
    print('  If they concentrate on badly-timed frames, the prior is dragging')
    print('  predictions off-target and the tracking gain is suspect.\n')
    for arm, D in (('baseline', B), ('C2', C)):
        fd = np.concatenate([D[k]['frame_diff'] for k in keys]) / FPS
        if 'cls_1' not in D[keys[0]]:
            print('  no bar class recorded'); break
        bar = np.concatenate([D[k]['cls_1'] for k in keys])
        n = min(len(fd), len(bar)); fd, bar = fd[:n], bar[:n]
        well, bad_t = fd <= THRESH, fd > THRESH
        print(f'  {arm:8s}  bar acc | well-timed (<={THRESH}s) = '
              f'{100*bar[well].mean():5.1f}%  (n={well.sum()})'
              f'   | badly-timed = {100*bar[bad_t].mean():5.1f}%  (n={bad_t.sum()})')

    # the decisive number: of frames C2 times WELL, how often is the bar wrong,
    # and how does that compare with the baseline on the same frames?
    fdB = np.concatenate([B[k]['frame_diff'] for k in keys]) / FPS
    fdC = np.concatenate([C[k]['frame_diff'] for k in keys]) / FPS
    if 'cls_1' in B[keys[0]] and 'cls_1' in C[keys[0]]:
        barB = np.concatenate([B[k]['cls_1'] for k in keys])
        barC = np.concatenate([C[k]['cls_1'] for k in keys])
        n = min(len(fdB), len(fdC), len(barB), len(barC))
        fdB, fdC, barB, barC = fdB[:n], fdC[:n], barB[:n], barC[:n]
        flipped = (barB == 1) & (barC == 0)          # C2 broke the bar
        print(f'\n  frames where C2 broke a correct bar: {int(flipped.sum())} '
              f'({100*flipped.mean():.1f}% of all frames)')
        if flipped.sum():
            print(f'    of those, C2 still times <= {THRESH}s: '
                  f'{100*np.mean(fdC[flipped] <= THRESH):.1f}%')
            print(f'    median timing error on those frames: '
                  f'baseline {np.median(fdB[flipped]):.3f}s -> C2 {np.median(fdC[flipped]):.3f}s')
        gained = (fdB > THRESH) & (fdC <= THRESH)
        lost = (fdB <= THRESH) & (fdC > THRESH)
        print(f'  frames C2 gained: {int(gained.sum())}   lost: {int(lost.sum())}   '
              f'net: {int(gained.sum() - lost.sum())}')


if __name__ == '__main__':
    main()
