"""Rank existing selector checkpoints on a dump, offline, in seconds.

Selecting a variant currently costs one eight-minute eval.py run each. The
rollout in the trainer is the same loop the decoder runs, so pointing it at a
dump ranks every checkpoint at once -- which is what makes a harder validation
set worth building at all, since the whole point is to re-rank cheaply.
"""
import argparse
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.train_cand_scorer import load_dumps, rollout
from extensions.heads.cand_scorer import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', nargs='+', required=True)
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--th', type=float, default=10.0)
    a = ap.parse_args()
    pieces = load_dumps(sorted(sum([glob.glob(p) for p in a.dump], [])))
    n = sum(len(p['cand']) for p in pieces)
    print(f'{len(pieces)} pages, {n} frames\n')
    rows = []
    for c in sorted(sum([glob.glob(p) for p in a.ckpts], [])):
        try:
            m, meta = load(c)
        except Exception as e:
            print(f'  {c.split("/")[-1]:<16} FAILED to load: {e}')
            continue
        r, arg, orc, nn = rollout(m, pieces, m.use_abs_obj, th=a.th)
        rows.append((r, c.split('/')[-1][:-3], m.nf, m.featdim, m.featproj,
                     m.n_params, arg, orc))
    rows.sort(reverse=True)
    print(f'{"variant":<14}{"nf":>4}{"fdim":>6}{"proj":>6}{"params":>9}'
          f'{"rollout":>9}{"headroom":>10}')
    for r, name, nf, fd, fp, np_, arg, orc in rows:
        h = 100 * (r - arg) / max(orc - arg, 1e-9)
        print(f'{name:<14}{nf:>4}{fd:>6}{fp:>6}{np_:>9,}{r:>9.2f}{h:>9.1f}%')
    if rows:
        print(f'\n  detector argmax on this dump: {rows[0][6]:.2f}   '
              f'oracle {rows[0][7]:.2f}')
        print('  (room argmax is 80.0 -- the closer this is, the more the '
              'ranking here can be trusted)')


if __name__ == '__main__':
    main()
