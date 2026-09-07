"""Rank every scorer on the held-out split, which shares no piece with room.

`do` has been the selector so far and it is a weak one: it correlates with
room at Spearman +0.407, and it shares all 16 pieces with room, so only the
acoustic condition differs and piece identity leaks. A selector that picks the
right model for the wrong reason will keep doing so.

The held-out split shares no piece with room OR with training, and has 80
pieces against room's 16. If its ranking predicts room's, it is the selection
instrument this project has been missing, and the choice of vel_p8 stops
resting on a proxy that leaks.

Run on snr12 and snr6: snr12 sits nearest room in difficulty (86.8 against
91.4) and snr6 is well clear of the ceiling, so agreement between the two is
itself evidence the ranking is about the model rather than the tier.
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load as load_ckpt

M = '/scratch/pmohseni/omr/scorer'
MODELS = ([('ir_only', f'{M}/ir_only.pt'), ('noz_only', f'{M}/noz_only.pt')]
          + [(n, f'{M}/{n}.pt') for n in
             ('feat_base', 'feat_small', 'feat_wide', 'vel_feat', 'vel_only')]
          + [(n, f'{M}/grid/{n}.pt') for n in
             ('vel_p8', 'vel_p32', 'vel_p64', 'novel_p8', 'novel_p32', 'novel_p64')])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiers', default='12,6')
    a = ap.parse_args()

    have = [(n, p) for n, p in MODELS if glob.glob(p)]
    print(f'{len(have)} checkpoints\n')
    scores = {}
    for tier in a.tiers.split(','):
        pages = load_with_feat(f'/scratch/pmohseni/omr/candhv/valid_snr{tier}.npz')
        for n, p in have:
            m = load_ckpt(p)[0]
            acc, _ = rollout(m, pages, blend=0.7)
            scores.setdefault(n, {})[tier] = acc
            print(f'  snr{tier:<4s} {n:12s} {acc:6.2f}', flush=True)
        del pages

    tiers = a.tiers.split(',')
    print(f'\n{"model":12s} ' + ' '.join(f'{"snr" + t:>8s}' for t in tiers))
    order = sorted(have, key=lambda x: -scores[x[0]][tiers[0]])
    for n, _ in order:
        print(f'{n:12s} ' + ' '.join(f'{scores[n][t]:8.2f}' for t in tiers))
    if len(tiers) > 1:
        v = [[scores[n][t] for n, _ in have] for t in tiers]
        print(f'\nSpearman between tiers: {spearman(v[0], v[1]):+.3f} '
              f'(high means the ranking is about the model, not the tier)')
    print(f'\nheld-out picks: {order[0][0]}')


if __name__ == '__main__':
    main()
