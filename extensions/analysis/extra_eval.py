"""Extra results for the paper, each readable on its own.

  detectors   argmax / prior / scorer transferred from cyolo_sb / scorer
              retrained on the detector's own candidates, for every released
              CYOLO variant, on the real recordings
  nohist      the scorer without history features, real + held-out
  budget      accuracy and oracle coverage against the candidate budget K
  staff       how often the selected note is on the right staff
  cost        wall time of one decoder step (features + scorer), one thread
Sections whose inputs do not exist yet are skipped, so this can be run early.
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import rollout_hits
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand
from extensions.analysis.offline_decode import TH
from extensions.heads.cand_scorer import load as load_ckpt

O = '/scratch/pmohseni/omr'
M = f'{O}/scorer'
SHIP = f'{M}/nbr/nbrp64_s0.pt'
ROOM = f'{O}/candf256/room.npz'
HV = f'{O}/candhv256/valid_snr12.npz'


def acc(model, pages, **kw):
    hits, _ = rollout_hits(model, pages, **kw)
    return 100.0 * hits.mean()


def seeds(pattern, pages):
    return [acc(load_ckpt(p)[0], pages) for p in sorted(glob.glob(pattern))]


def fmt(v):
    if not v:
        return '   n/a'
    v = np.array(v)
    return f'{v.mean():6.2f} +- {v.std(ddof=1) if len(v) > 1 else 0:4.2f} [{len(v)}]'


def detectors(ship):
    print('#### detectors (real recordings, 0.5 s)')
    print(f'{"detector":12s} {"argmax":>7s} {"prior":>7s} {"transfer":>9s}   retrained')
    for det, dump in (('cyolo_sb', ROOM), ('cyolo', f'{O}/cand_cyolo/room.npz'),
                      ('cyolo_sb_a', f'{O}/cand_cyolo_sb_a/room.npz')):
        if not os.path.exists(dump):
            print(f'{det:12s} (no dump yet)')
            continue
        pages = load_with_feat(dump)
        a, _ = rollout_argmax(pages)
        h, _ = rollout_hand(pages)
        own = (seeds(f'{M}/nbr/nbrp64_s[0-5].pt', pages) if det == 'cyolo_sb'
               else seeds(f'{M}/extra/{det}_s[0-9].pt', pages))
        print(f'{det:12s} {100 * a.mean():7.2f} {100 * h.mean():7.2f} '
              f'{acc(ship, pages):9.2f}   {fmt(own)}', flush=True)


def nohist():
    pat = f'{M}/extra/nohist_s[0-9].pt'
    if not glob.glob(pat):
        print('#### nohist: no checkpoints yet')
        return
    print('#### history-free scorer vs full (same recipe, seeds 0-2)')
    for tag, dump in (('real', ROOM), ('held-out 12 dB', HV)):
        pages = load_with_feat(dump)
        print(f'{tag:15s} nohist {fmt(seeds(pat, pages))}   '
              f'full {fmt(seeds(f"{M}/nbr/nbrp64_s[0-2].pt", pages))}', flush=True)
        del pages


def budget(ship, pages):
    print('#### candidate budget K (real recordings)')
    print(f'{"K":>4s} {"covered":>8s} {"scorer":>7s}')
    for k in (4, 8, 16, 32, 64, 128, 256):
        cov = n = 0
        for p in pages:
            for i, c in enumerate(p['cand']):
                if c.shape[0] == 0:
                    continue
                n += 1
                cov += bool((np.abs(c[:k, 5] - p['t_gt'][i]) <= TH).any())
        print(f'{k:4d} {100.0 * cov / n:8.2f} {acc(ship, pages, topk=k):7.2f}', flush=True)


def staff():
    from extensions.analysis.musical_cases import load_traj
    print('#### selected note on the correct staff (real recordings)')
    for n in ('baseline_room', 'selected_room', 'nbrp64s0_room'):
        t = load_traj(f'{O}/traj/{n}.traj.npz')
        sp = np.concatenate([d['staff_pred'] for d in t.values()])
        sg = np.concatenate([d['staff_gt'] for d in t.values()])
        print(f'{n:15s} {100.0 * np.mean(sp == sg):6.2f}')


def cost(ship, pages):
    t0 = time.perf_counter()
    hits, _ = rollout_hits(ship, pages)
    dt = time.perf_counter() - t0
    print(f'#### cost: {1000 * dt / len(hits):.2f} ms per decoder step '
          f'(features + scorer over 256 candidates, one CPU thread, {len(hits)} steps)')


def main():
    ship = load_ckpt(SHIP)[0]
    detectors(ship)
    room = load_with_feat(ROOM)
    budget(ship, room)
    staff()
    cost(ship, room)
    del room
    nohist()


if __name__ == '__main__':
    main()
