"""Is C2's hand-picked motion model anywhere near the truth?

C2 scores a candidate by log p_obj + lam * log P(displacement), with

    fwd_px   = 6.0     "how far the player moves per frame"
    sigma_px = 18.0    "how much that varies"

Both were guessed, never fitted, and both are GLOBAL CONSTANTS -- one number for
every piece at every tempo, on a corpus that runs from a slow Chopin nocturne to
a Bach invention. If the true per-frame displacement varies substantially across
pieces, then a single fwd_px is systematically wrong on most of them and an
online estimate of the local speed should beat it.

This measures the truth directly from the TRAINING split's ground truth -- no
detector involved, and no contact with the room test set. Unrolling follows
data_utils.py:125-137 exactly (sort staves by y, cumsum of per-staff max x),
because a displacement measured in any other coordinate is not the quantity the
decoder actually conditions on.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

FPS = 22050 / 1102.5      # SAMPLE_RATE / HOP_SIZE, data_utils.py:11-14


def unrolled(coords):
    """-> (x_unrolled, onset_seconds) per note, unrolled staff by staff.

    Group by system_idx, NOT note_y. note_y is the individual note's height on
    the staff -- its pitch -- so one system yields as many distinct note_y
    values as it has pitches. Grouping on it invents dozens of phantom staves
    and the cumulative offset explodes: it reported a median speed of 100
    px/frame, i.e. 2000 px/s across a page 835 px wide.
    """
    xs, ons = [], []
    pages = sorted({c['page_nr'] for c in coords})
    for pg in pages:
        pc = [c for c in coords if c['page_nr'] == pg]
        sids = sorted({float(c['system_idx']) for c in pc})
        max_xes = [0.0]
        for s in sids:
            max_xes.append(max(float(c['note_x']) for c in pc if float(c['system_idx']) == s))
        add = np.cumsum(max_xes)[:-1]
        for i, s in enumerate(sids):
            for c in pc:
                if float(c['system_idx']) == s:
                    xs.append(float(c['note_x']) + add[i])
                    ons.append(float(c['onset']))
    o = np.argsort(ons)
    return np.array(xs)[o], np.array(ons)[o]


def main():
    files = sorted(glob.glob('/scratch/pmohseni/datasets/cyolo_data/msmd/msmd_train/*.npz'))
    print(f'{len(files)} training pieces (test set untouched)\n')

    per_piece, all_v = [], []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            x, on = unrolled(list(z['coords']))
        except Exception:
            continue
        fr = on * FPS
        dx, df = np.diff(x), np.diff(fr)
        ok = (df > 0) & (dx >= 0)          # same page, forward in time
        if ok.sum() < 20:
            continue
        v = dx[ok] / df[ok]                # px per frame
        v = v[v < 200]                     # drop page/staff wraps
        if v.size < 20:
            continue
        per_piece.append((f.split('/')[-1][:34], float(np.median(v)), v.size))
        all_v.append(v)

    v = np.concatenate(all_v)
    med = np.array([p[1] for p in per_piece])
    print(f'pieces measured: {len(per_piece)}   note transitions: {v.size}\n')
    print('TRUE per-frame displacement, pooled over all pieces (px/frame):')
    for q in (5, 25, 50, 75, 95):
        print(f'   p{q:<2d} = {np.percentile(v, q):7.2f}')
    print(f'   mean = {v.mean():.2f}   std = {v.std():.2f}')
    print(f'\nC2 assumes fwd_px = 6.00, sigma_px = 18.00')
    print(f'   truth: median {np.median(v):.2f}, IQR '
          f'{np.percentile(v,25):.2f}-{np.percentile(v,75):.2f}')

    print('\nPER-PIECE median speed -- this is what a global constant must cover:')
    for q in (5, 25, 50, 75, 95):
        print(f'   p{q:<2d} = {np.percentile(med, q):7.2f}')
    print(f'   min {med.min():.2f}   max {med.max():.2f}   '
          f'ratio max/min = {med.max()/max(med.min(),1e-6):.1f}x')
    sl = sorted(per_piece, key=lambda r: r[1])
    print(f'   slowest: {sl[0][0]} {sl[0][1]:.2f} px/frame')
    print(f'   fastest: {sl[-1][0]} {sl[-1][1]:.2f} px/frame')

    # How much of the spread is BETWEEN pieces vs WITHIN a piece? If most of it
    # is between, a per-piece (online) estimate captures most of the headroom;
    # if most is within, the prior must stay broad no matter what.
    within = np.mean([np.var(a[a < 200]) for a in all_v if (a < 200).sum() > 20])
    between = np.var(med)
    print(f'\nvariance between pieces = {between:.2f}   within pieces = {within:.2f}')
    print(f'   -> {100*between/(between+within):.0f}% of the variance is BETWEEN pieces')


if __name__ == '__main__':
    main()
