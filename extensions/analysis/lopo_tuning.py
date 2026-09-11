"""Leave-one-piece-out selection of the decoder's four scalars on real audio.

The prior (mu, sigma, J) and the blend weight beta are four scalars with no
disjoint real-audio development set to tune them on: the only real recordings
are the 16 test pieces, and synthetic proxies do not select beta (validation
picks 1.0 -> room 85.20, held-out picks 0.9 -> 88.29, while room is sharply
peaked at 0.7). The standard answer on a small set is nested cross-validation:
for each piece, choose the four scalars on the OTHER 15 pieces and score the
held-out piece with them. No piece's own labels ever choose its constants, and
the scorer's weights never see real audio at all.

Per fold the prior is chosen with the hand decoder over the full grid, then
beta for the reported scorer at that prior -- the same two-stage procedure the
proxy tuning used -- and the pooled onset accuracy is reported for both.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import paired_bootstrap, piece_of, rollout_hits
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand
from extensions.heads.cand_scorer import load as load_ckpt

ROOM = '/scratch/pmohseni/omr/candf256/room.npz'
SHIP = '/scratch/pmohseni/omr/scorer/nbr/nbrp64_s0.pt'
SHIPPED = (6.0, 18.0, -6.0)
BLENDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID = list(itertools.product((3.0, 4.5, 6.0, 8.0, 10.0), (12.0, 18.0, 27.0),
                              (-4.0, -6.0, -8.0)))


def main():
    room = load_with_feat(ROOM)
    arg, pg = rollout_argmax(room)
    pieces = piece_of(pg)
    names = sorted(set(pieces))
    print(f'{len(arg)} onsets, {len(names)} pieces; argmax {100 * arg.mean():.2f}', flush=True)

    hand = {}
    for pr in GRID:
        h, _ = rollout_hand(room, fwd=pr[0], sigma=pr[1], jump=pr[2])
        hand[pr] = h
    fold_prior = {p: max(GRID, key=lambda k: hand[k][pieces != p].mean()) for p in names}

    m = load_ckpt(SHIP)[0]
    full = {}
    for pr in sorted(set(fold_prior.values()) | {SHIPPED}):
        for b in BLENDS:
            s, _ = rollout_hits(m, room, blend=b, fwd=pr[0], sigma=pr[1], jump=pr[2])
            full[(pr, b)] = s
            print(f'  prior {pr} blend {b:.1f}: {100 * s.mean():6.2f}', flush=True)

    lo_hand = np.zeros(len(pieces), bool)
    lo_full = np.zeros(len(pieces), bool)
    print(f'\n{"held-out piece":52s} {"prior":>16s} {"beta":>5s}')
    for p in names:
        held = pieces == p
        pr = fold_prior[p]
        b = max(BLENDS, key=lambda v: full[(pr, v)][~held].mean())
        lo_hand[held] = hand[pr][held]
        lo_full[held] = full[(pr, b)][held]
        print(f'{p[:52]:52s} {str(pr):>16s} {b:5.1f}')

    print(f'\nleave-one-piece-out, pooled over {len(pieces)} onsets:')
    print(f'  prior only   {100 * lo_hand.mean():6.2f}')
    print(f'  full decoder {100 * lo_full.mean():6.2f}')
    for tag, h in (('argmax -> prior', lo_hand), ('argmax -> full', lo_full)):
        mu, lo, hi, pv = paired_bootstrap(arg, h, pieces)
        print(f'  {tag:16s} {mu:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  p={pv:.4f}')
    mu, lo, hi, pv = paired_bootstrap(lo_hand, lo_full, pieces)
    print(f'  {"prior -> full":16s} {mu:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  p={pv:.4f}')


if __name__ == '__main__':
    main()
