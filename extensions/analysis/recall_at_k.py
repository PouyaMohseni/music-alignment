"""How many candidates does the decision actually need?

The oracle argument says the answer is in the set. It does not say how big the
set has to be, and K=256 was inherited rather than justified. This measures
both curves against K: the fraction of onsets where SOME candidate in the top K
lies within the tolerance, which upper-bounds any selector, and the causal
monotone oracle, which must commit at each instant and never move backwards.
The gap between them is what a real-time constraint costs; the gap between the
causal oracle and the deployed rule is what the decision costs.
"""
from __future__ import annotations
import argparse, sys
import numpy as np
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.offline_decode import TH

KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/scratch/pmohseni/omr/candf256/room.npz')
    a = ap.parse_args()
    pages = load_with_geo = load_with_feat(a.dump)

    tot = 0
    avail = {k: 0 for k in KS}          # any candidate in top-K within tolerance
    caus = {k: 0 for k in KS}           # causal, monotone oracle over top-K
    argm = 0
    for p in pages:
        prev = {k: None for k in KS}
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            tot += 1
            gt = float(p['t_gt'][i])
            ok = np.abs(c[:, 5] - gt) <= TH
            if ok[0]:
                argm += 1
            for k in KS:
                sub = ok[:k]
                if sub.any():
                    avail[k] += 1
                # causal monotone: only candidates at or ahead of the last
                # committed position are reachable, and the oracle takes a
                # correct one when one exists there
                x = c[:k, 0]
                reach = np.ones(len(x), bool) if prev[k] is None else (x >= prev[k] - 1e-6)
                good = np.flatnonzero(sub & reach)
                if good.size:
                    caus[k] += 1
                    # take the EARLIEST correct candidate, not the most
                    # confident one: under a monotone constraint every pixel
                    # committed now is reachability given up later, and a
                    # greedy oracle that grabs a far-forward correct box
                    # strands the onsets behind it
                    prev[k] = float(x[good].min())
                elif reach.any():
                    prev[k] = float(x[reach].min())
    print(f'{tot} scored onsets, {len(pages)} pages, tolerance {TH} frames\n')
    print(f'{"K":>5} {"candidate available (%)":>24} {"causal monotone oracle (%)":>28}')
    for k in KS:
        print(f'{k:5d} {100.0*avail[k]/tot:24.2f} {100.0*caus[k]/tot:28.2f}')
    print(f'\ndeployed rule (K=1 by construction, argmax): {100.0*argm/tot:.2f}')


if __name__ == '__main__':
    main()
