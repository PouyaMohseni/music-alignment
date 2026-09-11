"""Score an every-frame trajectory two ways: all frames, and onset frames only.

Under --only_onsets, cyolo's load_dataset drops every frame that is not an
annotated note onset, so the tracker is only ever run at the moments notes
start, and a decoder that carries state from one step to the next is handed
the annotated inter-onset gaps. Run WITHOUT the flag and every frame is
loaded, decoded and scored. The onset subset of that same run is then the
number to set against the onset-only protocol, with nothing about WHEN notes
start given to the tracker: it stepped through every frame at 20 fps.

Onset frames are recovered exactly as cyolo's load_piece defines them --
int(onset * FPS) per annotated note, is_onset = frame in onsets -- so the
onset count must come back as 4149, the onset-only protocol's own count.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.musical_cases import FPS, load_piece, load_traj, merge_pages

TH = (0.05, 0.1, 0.5, 1.0, 5.0)


def pct(err):
    return '  '.join(f'{100.0 * np.mean(err <= t):5.1f}' for t in TH)


def main():
    for path in sys.argv[1:]:
        traj = load_traj(path)
        alls, ons, rows = [], [], []
        for pn in sorted({p.rsplit('_page_', 1)[0] for p in traj}):
            tr = merge_pages(traj, pn)
            pc = load_piece(pn.replace('_room', ''), 'room')
            onsets = np.unique([int(float(c['onset']) * FPS) for c in pc['coords']])
            err = np.abs(tr['t_pred'] - tr['t_gt']) / FPS
            on = np.isin(tr['frame'], onsets)
            alls.append(err)
            ons.append(err[on])
            rows.append((pn, len(err), int(on.sum()), 100.0 * np.mean(err <= .5),
                         100.0 * np.mean(err[on] <= .5) if on.any() else float('nan')))
        a, o = np.concatenate(alls), np.concatenate(ons)
        print(f'##### {path.split("/")[-1]}')
        print(f'{"":22s} {"frames":>7s}   ' + '  '.join(f'{t:>5}' for t in TH))
        print(f'{"every frame":22s} {len(a):7d}   {pct(a)}')
        print(f'{"onset frames only":22s} {len(o):7d}   {pct(o)}')
        print(f'\n{"piece":52s} {"frames":>7s} {"onsets":>7s} {"all@.5":>7s} {"ons@.5":>7s}')
        for pn, n, k, x, y in rows:
            print(f'{pn[:52]:52s} {n:7d} {k:7d} {x:7.1f} {y:7.1f}')


if __name__ == '__main__':
    main()
