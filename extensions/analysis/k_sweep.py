"""CANDOR and the causal oracle as a function of the candidate-set size K.

The paper justifies K=256 from availability and the oracle alone, which shows
where the SET stops binding but not where the DECISION stops improving. Those
are different questions: a larger set can raise the ceiling and lower the
realised accuracy at the same time, because every extra candidate is one more
distractor the scorer has to rank below the right one.

Everything is decoded from the same dump with the list truncated to the K most
confident, so the detector is untouched and only the size of the set the
decision sees changes. The scorer was trained at K=256; evaluating it at
smaller K is a decoding-time change, and the set features it reads (the
pooled context, the crowding terms) are recomputed at each K rather than
carried over, so each row is internally consistent.

The LOPO protocol is the one in lopo_thresholds.py, refitted at every K: the
constants for a piece are chosen on the other 15 AT THAT K, so a row is not
K=256's constants applied to a smaller set.
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import piece_of
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand
from extensions.analysis.lopo_thresholds import BLENDS, GRID, errors
from extensions.heads.cand_scorer import load as load_ckpt

FPS = 22050 / 1102


def oracle_causal(pages, k, tol_s=0.5):
    """Best reachable accuracy for a rule that commits and never goes back.

    At each onset take the EARLIEST correct candidate at or after the previous
    commitment, not the most confident one: a greedy causal rule that jumps
    ahead to a later correct candidate strands itself for the onsets in
    between, so the earliest choice is the one that bounds the family.
    """
    tol = tol_s * FPS
    hit = tot = 0
    for p in pages:
        prev = -np.inf
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:k]
            # reachability is on the PAGE COORDINATE, column 0, because the
            # constraint being modelled is that the tracker never moves
            # backwards on the page. Testing it on the mapped time in column 5
            # is more permissive and inflates the oracle by three points at
            # K=64. Correctness is still judged on the mapped time.
            x, t = cs[:, 0], float(p['t_gt'][i])
            reach = x >= prev - 1e-6
            good = np.abs(cs[:, 5] - t) <= tol
            tot += 1
            ok = np.flatnonzero(good & reach)
            if ok.size:
                hit += 1
                prev = float(x[ok].min())
            elif reach.any():
                prev = float(x[reach].min())
    return 100.0 * hit / max(tot, 1)


def availability(pages, k, tol_s=0.5):
    tol = tol_s * FPS
    hit = tot = 0
    for p in pages:
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            tot += 1
            hit += int(np.any(np.abs(c[:k, 5] - float(p['t_gt'][i])) <= tol))
    return 100.0 * hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--ckpt', required=True, help='glob over seeds')
    ap.add_argument('--ks', default='32,64,128,256')
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    pages = load_with_feat(a.dump)
    ks = [int(v) for v in a.ks.split(',')]
    models = [load_ckpt(c)[0] for c in sorted(glob.glob(a.ckpt))]
    print(f'{a.label or a.dump}   {len(models)} seeds\n')
    print(f'{"K":>5s} {"avail":>7s} {"oracle":>7s} {"argmax":>7s} '
          f'{"+prior":>7s} {"CANDOR":>8s}')

    for k in ks:
        arg, pg = rollout_argmax(pages, topk=k)
        pieces = piece_of(pg)
        names = sorted(set(pieces))

        def macro(hits, held):
            return np.mean([hits[pieces == q].mean() for q in names if q != held])

        hand = {pr: rollout_hand(pages, fwd=pr[0], sigma=pr[1], jump=pr[2],
                                 topk=k)[0] for pr in GRID}
        fold = {p: max(GRID, key=lambda g: macro(hand[g], p)) for p in names}
        lo = np.zeros(len(pieces), bool)
        for p in names:
            lo[pieces == p] = hand[fold[p]][pieces == p]

        per_seed = []
        for m in models:
            cache = {}
            for pr in sorted(set(fold.values())):
                for b in BLENDS:
                    cache[(pr, b)] = errors(m, pages, b, pr, topk=k)
            picked = np.zeros(len(pieces))
            for p in names:
                pr = fold[p]
                b = max(BLENDS, key=lambda v: macro(cache[(pr, v)] <= 0.5, p))
                picked[pieces == p] = cache[(pr, b)][pieces == p]
            per_seed.append(100.0 * np.mean(picked <= 0.5))

        print(f'{k:5d} {availability(pages, k):7.2f} {oracle_causal(pages, k):7.2f} '
              f'{100.0 * arg.mean():7.2f} {100.0 * lo.mean():7.2f} '
              f'{np.mean(per_seed):8.2f}', flush=True)


if __name__ == '__main__':
    main()
