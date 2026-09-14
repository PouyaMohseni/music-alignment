"""Piece-level bootstrap interval on the headline, under the LOPO protocol.

Table 1 reports a point estimate on 16 pieces and a reviewer is entitled to ask
what 94.4 means when the evaluation set is that small.  This runs exactly the
protocol of lopo_thresholds.py -- the four constants chosen on the other 15
pieces, the sixteenth decoded with them -- and then resamples PIECES with
replacement, so the interval reflects the sampling unit that actually varies.

Two intervals are reported.  The first is on the accuracy itself.  The second
is on the gain over argmax, computed on the same resample, so it is paired and
does not double-count the between-piece variance the two arms share.
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


def boot(hits, pieces, n_boot=20000, seed=0):
    """Resample pieces with replacement; return mean and the 95% interval."""
    rng = np.random.default_rng(seed)
    names = np.unique(pieces)
    idx = {n: np.flatnonzero(pieces == n) for n in names}
    draws = []
    for _ in range(n_boot):
        take = np.concatenate([idx[n] for n in rng.choice(names, len(names), True)])
        draws.append(100.0 * hits[take].mean())
    d = np.asarray(draws)
    return 100.0 * hits.mean(), *np.percentile(d, [2.5, 97.5])


def boot_paired(hb, hc, pieces, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    names = np.unique(pieces)
    idx = {n: np.flatnonzero(pieces == n) for n in names}
    draws = []
    for _ in range(n_boot):
        take = np.concatenate([idx[n] for n in rng.choice(names, len(names), True)])
        draws.append(100.0 * (hc[take].mean() - hb[take].mean()))
    d = np.asarray(draws)
    return 100.0 * (hc.mean() - hb.mean()), *np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--ckpt', required=True, help='glob over seeds')
    ap.add_argument('--label', default='')
    ap.add_argument('--th', type=float, default=0.5)
    a = ap.parse_args()

    pages = load_with_feat(a.dump)
    arg, pg = rollout_argmax(pages)
    pieces = piece_of(pg)
    names = sorted(set(pieces))

    def macro(hits, held):
        return np.mean([hits[pieces == q].mean() for q in names if q != held])

    hand = {pr: rollout_hand(pages, fwd=pr[0], sigma=pr[1], jump=pr[2])[0]
            for pr in GRID}
    fold_prior = {p: max(GRID, key=lambda k: macro(hand[k], p)) for p in names}

    per_seed = []
    for ck in sorted(glob.glob(a.ckpt)):
        model = load_ckpt(ck)[0]
        cache = {}
        for pr in sorted(set(fold_prior.values())):
            for b in BLENDS:
                cache[(pr, b)] = errors(model, pages, b, pr)
        picked = np.zeros(len(pieces))
        for p in names:
            pr = fold_prior[p]
            b = max(BLENDS, key=lambda v: macro(cache[(pr, v)] <= a.th, p))
            picked[pieces == p] = cache[(pr, b)][pieces == p]
        per_seed.append((picked <= a.th).astype(np.float64))
        print(f'  seed {ck.split("/")[-1]:24s} {100.0 * per_seed[-1].mean():.2f}',
              flush=True)

    # the seed mean per onset, so the interval is on the number the table prints
    # rather than on one arbitrary seed
    hits = np.mean(per_seed, axis=0)
    argf = arg.astype(np.float64)

    print(f'\n{a.label or a.dump}   {len(hits)} onsets / {len(names)} pieces '
          f'/ {len(per_seed)} seeds, threshold {a.th}\n')
    m, lo, hi = boot(hits, pieces)
    print(f'CANDOR          {m:5.2f}   95% CI [{lo:.2f}, {hi:.2f}]')
    m, lo, hi = boot(argf, pieces)
    print(f'argmax          {m:5.2f}   95% CI [{lo:.2f}, {hi:.2f}]')
    m, lo, hi = boot_paired(argf, hits, pieces)
    print(f'gain (paired)   {m:+5.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]')

    print('\nper piece')
    for p in names:
        k = pieces == p
        print(f'  {p[:48]:48s} {int(k.sum()):5d} {100.0 * argf[k].mean():6.1f}'
              f' {100.0 * hits[k].mean():6.1f}')


if __name__ == '__main__':
    main()
