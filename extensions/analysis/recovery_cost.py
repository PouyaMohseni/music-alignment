"""What a recovery trigger costs when nothing goes wrong.

The discontinuity table measures the triggers only on spliced takes, where they
repair most of the damage the transition model does at a jump. That says
nothing about whether they belong in the deployed decoder, because a trigger
that fires spuriously during ordinary playing would trade a rare failure for a
common one. This runs the identical decoder, with and without each trigger, on
the UNSPLICED recordings under the same leave-one-piece-out protocol, so the
two halves of the decision can be read off one table.

A trigger is worth shipping if its cost here is inside the seed deviation and
its gain on Table 3 is not.
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
from extensions.analysis.heldout_compare import piece_of
from extensions.analysis.jump_eval import FPS, rollout
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand

GRID = list(itertools.product((3.0, 4.5, 6.0, 8.0, 10.0), (12.0, 18.0, 27.0),
                              (-4.0, -6.0, -8.0)))
BLENDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TH = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--ckpt', required=True, help='glob over seeds')
    ap.add_argument('--modes', nargs='+', default=['none', 'conf', 'beam'])
    ap.add_argument('--label', default='')
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
    print(f'{a.label or a.dump}   {len(arg)} onsets / {len(names)} pieces   '
          f'argmax {100 * arg.mean():.2f}\n', flush=True)
    print(f'{"mode":10s} ' + '  '.join(f'{c.split("/")[-1][:14]:>14s}'
                                       for c in sorted(glob.glob(a.ckpt))) + '    mean')

    from extensions.heads.cand_scorer import load as load_ckpt
    for mode in a.modes:
        per = []
        for ck in sorted(glob.glob(a.ckpt)):
            model = load_ckpt(ck)[0]
            # one pass per (prior, blend) is too slow with a trigger in the
            # loop, so the scalars are fixed to the fold choice at blend 0.7,
            # the value the sweep picks for this recipe, and only the trigger
            # varies. Both columns therefore use identical constants.
            hits = np.zeros(len(pieces), bool)
            off = 0
            for p in pages:
                pr = fold_prior[piece_of([p['name']] * 1)[0]]
                pred = rollout(model, p, None, mode=mode, blend=0.7,
                               fwd=pr[0], sigma=pr[1], jump=pr[2])
                gt = np.asarray(p['t_gt'], float)
                ok = np.abs(pred - gt) / FPS <= TH
                hits[off:off + len(ok)] = ok
                off += len(ok)
            per.append(100.0 * hits.mean())
        print(f'{mode:10s} ' + '  '.join(f'{v:14.2f}' for v in per)
              + f'  {np.mean(per):6.2f}', flush=True)


if __name__ == '__main__':
    main()
