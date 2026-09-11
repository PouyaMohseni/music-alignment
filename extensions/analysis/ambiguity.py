"""Are the errors ambiguous? Compare the notes at the chosen and true positions.

For every scored onset outside the threshold: the chord struck at the predicted
time and the chord struck at the true time, both read from the performance's
note annotations (analysis only -- nothing here feeds a model). An identical
pitch set means one frame of the page genuinely cannot tell the two apart; no
shared pitch means the tracker is simply in the wrong place.

The demo page's figures (355 errors, 18 identical, 59.2% with nothing in
common, median overlap 0.000) came from an ad-hoc pass over the 91.4 selector's
trajectory that was never saved as a script. So pass that trajectory FIRST: if
this definition does not reproduce those figures on it, it is not the same
measurement and its output for a new model must not replace them.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.musical_cases import (FPS, TH_SEC, load_piece, load_traj,
                                               merge_pages)


def chords(coords, scale):
    """sorted onset times (seconds) and the pitch set struck at each."""
    by = {}
    for c in coords:
        by.setdefault(round(float(c['onset']) * scale, 3), set()).add(int(c['pitch']))
    t = np.array(sorted(by))
    return t, [frozenset(by[k]) for k in t]


def nearest(t, times):
    j = int(np.argmin(np.abs(times - t)))
    return j, abs(float(times[j]) - t)


def units(traj):
    """Annotated onsets in seconds or in frames? Whichever puts the true times
    on top of annotated onsets. Printed, never silently assumed."""
    lags = {1.0: [], 1.0 / FPS: []}
    for pn in sorted({p.rsplit('_page_', 1)[0] for p in traj})[:4]:
        pc = load_piece(pn.replace('_room', ''), 'room')
        tg = merge_pages(traj, pn)['t_gt'][:200] / FPS
        for s in lags:
            times, _ = chords(pc['coords'], s)
            lags[s] += [nearest(t, times)[1] for t in tg]
    med = {s: float(np.median(v)) for s, v in lags.items()}
    s = min(med, key=med.get)
    print(f'  onset units: {"seconds" if s == 1.0 else "frames"} '
          f'(median lag {1000 * med[s]:.1f} ms; other reading {1000 * max(med.values()):.1f} ms)')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traj', nargs='+')
    a = ap.parse_args()
    for path in a.traj:
        print(f'\n##### {path.split("/")[-1]}')
        traj = load_traj(path)
        scale = units(traj)
        ov, same, none, n = [], 0, 0, 0
        for pn in sorted({p.rsplit('_page_', 1)[0] for p in traj}):
            pc = load_piece(pn.replace('_room', ''), 'room')
            times, sets = chords(pc['coords'], scale)
            tr = merge_pages(traj, pn)
            err = np.abs(tr['t_pred'] - tr['t_gt']) / FPS
            n += len(err)
            bad = err > TH_SEC
            for tp, tg in zip(tr['t_pred'][bad] / FPS, tr['t_gt'][bad] / FPS):
                A = sets[nearest(tp, times)[0]]
                B = sets[nearest(tg, times)[0]]
                u = len(A | B)
                ov.append(len(A & B) / u if u else 0.0)
                same += int(A == B)
                none += int(not (A & B))
        ov = np.array(ov)
        k = len(ov)
        print(f'  {n} onsets, {k} outside threshold ({100.0 * (n - k) / n:.2f}% hit)')
        print(f'  identical pitch sets    {same:5d}  {100.0 * same / k:5.1f}%')
        print(f'  no pitch in common      {none:5d}  {100.0 * none / k:5.1f}%')
        print(f'  median overlap (Jaccard) {np.median(ov):.3f}  mean {ov.mean():.3f}')


if __name__ == '__main__':
    main()
