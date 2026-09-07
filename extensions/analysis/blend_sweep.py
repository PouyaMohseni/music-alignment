"""Is blend=0.7 right for vel_p8, or inherited from ir_only?

0.7 was chosen for the featureless selector and every feature model has been
evaluated at it since, including vel_p8. A model with a feature pathway has
strictly more information than the hand prior carries, so the weight on the
hand term is not obviously the same. Swept on `do`, read once on room.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool

import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load as load_ckpt

_G = {}


def _init(ckpt, dumps):
    _G['model'] = load_ckpt(ckpt)[0]
    for nm, d in dumps.items():
        _G[nm] = load_with_feat(d)


def _run(job):
    which, b = job
    acc, n = rollout(_G['model'], _G[which], blend=b)
    return which, b, acc, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    dumps = {'do': '/scratch/pmohseni/omr/candf/do.npz',
             'room': '/scratch/pmohseni/omr/candf/room.npz'}
    blends = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with Pool(a.procs, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        res = pool.map(_run, [('do', b) for b in blends]
                       + [('room', b) for b in blends])
    do = {b: acc for w, b, acc, _ in res if w == 'do'}
    rm = {b: acc for w, b, acc, _ in res if w == 'room'}
    print(f'{"blend":>6s} {"do":>7s} {"room":>7s}')
    for b in blends:
        mark = '   <- shipped' if b == 0.7 else ''
        print(f'{b:6.1f} {do[b]:7.2f} {rm[b]:7.2f}{mark}')
    bb = max(do, key=do.get)
    print(f'\n`do` picks blend={bb} -> do {do[bb]:.2f}, room {rm[bb]:.2f}')
    print(f'shipped blend=0.7   -> do {do[0.7]:.2f}, room {rm[0.7]:.2f}')
    br = max(rm, key=rm.get)
    print(f'(best on room would be {br} at {rm[br]:.2f}, not a defensible pick)')


if __name__ == '__main__':
    main()
