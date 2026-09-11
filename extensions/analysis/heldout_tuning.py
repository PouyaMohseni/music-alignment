"""Choose the decoder's constants on data disjoint from the test pieces, then
read room once.

The transition prior (mu, sigma, J) was first swept on the room recordings, and
the blend weight was re-checked only on the direct-pickup takes of the same
performances. For the paper every constant has to come from data disjoint from
the test pieces. This picks the prior constants with the hand decoder and the
blend weight with the reported scorer on --tune (the 80 held-out pieces at
12 dB by default, or the MSMD validation split), then reports the room number
of the chosen setting next to the shipped one.

Held-out (2026-09-11) chose fwd 10 / sigma 18 / J -8 and blend 0.9, which reads
88.29 on room against the shipped 94.89 -- held-out is a known poor proxy for
room (the detector saw those pieces in training), so the validation split is
the standard alternative and is run as a second pass.

--sensitivity adds a room sweep over the blend weight at both priors. It is
ANALYSIS ONLY, printed under its own header, and never feeds the selection.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import rollout_hits
from extensions.analysis.ladder_ci import rollout_hand
from extensions.heads.cand_scorer import load as load_ckpt

HV = '/scratch/pmohseni/omr/candhv256/valid_snr12.npz'
ROOM = '/scratch/pmohseni/omr/candf256/room.npz'
SHIP = '/scratch/pmohseni/omr/scorer/nbr/nbrp64_s0.pt'
SHIPPED = (6.0, 18.0, -6.0)
BLENDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', default=HV, help='dump the constants are chosen on')
    ap.add_argument('--sensitivity', action='store_true')
    a = ap.parse_args()
    print(f'tuning set: {a.tune}', flush=True)

    gaps = []
    for f in sorted(glob.glob('/scratch/pmohseni/omr/candf/train_c*.npz')):
        z = np.load(f)
        for k in z.files:
            if k.endswith('||frame'):
                d = np.diff(z[k].astype(np.int64))
                gaps.append(d[d > 0])
    g = np.concatenate(gaps)
    print(f'training inter-onset gap: median {np.median(g):.1f} frames, '
          f'mean {g.mean():.2f}, n={len(g)}', flush=True)

    tune = load_with_feat(a.tune)
    grid = []
    for fwd, sig, jump in itertools.product((3.0, 4.5, 6.0, 8.0, 10.0),
                                            (12.0, 18.0, 27.0), (-4.0, -6.0, -8.0)):
        h, _ = rollout_hand(tune, fwd=fwd, sigma=sig, jump=jump)
        grid.append((100.0 * h.mean(), fwd, sig, jump))
        print(f'  hand fwd={fwd:4.1f} sigma={sig:4.1f} J={jump:4.1f}  {grid[-1][0]:6.2f}',
              flush=True)
    grid.sort(reverse=True)
    best = grid[0][1:]
    ship_h = next(r[0] for r in grid if r[1:] == SHIPPED)
    print(f'\ntuned prior: best fwd={best[0]} sigma={best[1]} J={best[2]} '
          f'-> {grid[0][0]:.2f}; shipped 6/18/-6 -> {ship_h:.2f}', flush=True)

    m = load_ckpt(SHIP)[0]
    settings = [('chosen', best)] + ([('shipped', SHIPPED)] if best != SHIPPED else [])
    acc = {}
    for tag, (fwd, sig, jump) in settings:
        for b in BLENDS:
            hits, _ = rollout_hits(m, tune, blend=b, fwd=fwd, sigma=sig, jump=jump)
            acc[(tag, b)] = 100.0 * hits.mean()
            print(f'  scorer {tag:7s} blend={b:.1f}  {acc[(tag, b)]:6.2f}', flush=True)
    b_star = max(BLENDS, key=lambda b: acc[('chosen', b)])
    print(f'\ntuned blend at the chosen prior: {b_star:.1f} '
          f'({acc[("chosen", b_star)]:.2f})', flush=True)
    del tune

    room = load_with_feat(ROOM)
    rows = [('chosen', best, b_star), ('shipped', SHIPPED, 0.7)]
    print('\n#### room, read once')
    for tag, (fwd, sig, jump), b in rows:
        h, _ = rollout_hand(room, fwd=fwd, sigma=sig, jump=jump)
        s, _ = rollout_hits(m, room, blend=b, fwd=fwd, sigma=sig, jump=jump)
        print(f'{tag:8s} prior {fwd}/{sig}/{jump} blend {b:.1f}:  '
              f'prior only {100 * h.mean():6.2f}   with scorer {100 * s.mean():6.2f}',
              flush=True)

    if a.sensitivity:
        print('\n#### room sensitivity (analysis only, not used for selection)')
        for tag, (fwd, sig, jump) in (('shipped', SHIPPED), ('chosen', best)):
            vals = []
            for b in BLENDS:
                s, _ = rollout_hits(m, room, blend=b, fwd=fwd, sigma=sig, jump=jump)
                vals.append(f'{b:.1f}:{100 * s.mean():.2f}')
            print(f'{tag:8s} prior {fwd}/{sig}/{jump}   ' + '  '.join(vals), flush=True)


if __name__ == '__main__':
    main()
