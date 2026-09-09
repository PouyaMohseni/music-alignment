"""Compare feature sets as CONFIGURATIONS, never as single checkpoints.

Every lesson of the last two days is in this file. One draw of a configuration
spans two points on room, so a single checkpoint proves nothing; the room test
set has sixteen independent pieces and cannot resolve two points either, nor
even measure seed spread (it reported 2.05 where the held-out set says 0.70).
So each configuration is summarised by its mean over seeds, the comparison that
decides anything is made on the held-out split, and room is reported alongside
without being allowed to choose.

Held-out is 80 pieces disjoint from room and from training, paired per onset
against the featureless selector.
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
from extensions.analysis.heldout_compare import (paired_bootstrap, piece_of,
                                                 rollout_hits)
from extensions.heads.cand_scorer import load as load_ckpt

M = '/scratch/pmohseni/omr/scorer'
CONFIGS = [
    ('vel_p8 (24 feat)', [f'{M}/grid/vel_p8.pt'] + sorted(glob.glob(f'{M}/seeds/vel_p8_s*.pt'))),
    ('nbr (33 feat)', sorted(glob.glob(f'{M}/nbr/nbr_s*.pt'))),
    ('tempo (37 feat)', sorted(glob.glob(f'{M}/tempo/tempo_s*.pt'))),
    ('dagger (33 feat, rollout states)', sorted(glob.glob(f'{M}/dagger/dagger_s*.pt'))),
    # without these two entries both directions train and then produce no
    # measurement at all, which is how an experiment quietly becomes a no-op
    ('dagrec w=0 (unwinnable states dropped)',
     sorted(glob.glob(f'{M}/dagrec/dagrec_w0.0_s*.pt'))),
    ('dagrec w=0.25', sorted(glob.glob(f'{M}/dagrec/dagrec_w0.25_s*.pt'))),
    ('crf (globally normalized)', sorted(glob.glob(f'{M}/crf/crf_s*.pt'))),
    ('pitch agreement (41 feat)', sorted(glob.glob(f'{M}/pitchsc/pitchsc_s*.pt'))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout',
                    default='/scratch/pmohseni/omr/candhv256/valid_snr12.npz')
    ap.add_argument('--room', default='/scratch/pmohseni/omr/candf256/room.npz')
    a = ap.parse_args()

    hv = load_with_feat(a.heldout)
    hb, pg = rollout_hits(load_ckpt(f'{M}/ir_only.pt')[0], hv)
    cl = piece_of(pg)
    room = load_with_feat(a.room)
    ir_room, _ = rollout(load_ckpt(f'{M}/ir_only.pt')[0], room, blend=0.7)
    print(f'held-out {len(hb)} onsets / {len(np.unique(cl))} pieces; '
          f'ir_only {100 * hb.mean():.2f}   room ir_only {ir_room:.2f}\n')

    summary = []
    for name, paths in CONFIGS:
        paths = [p for p in paths if glob.glob(p)]
        if not paths:
            print(f'--- {name}: no checkpoints yet\n')
            continue
        print(f'--- {name}  ({len(paths)} seeds)')
        hs, rs, ds = [], [], []
        for p in paths:
            m = load_ckpt(p)[0]
            h, _ = rollout_hits(m, hv)
            d, lo, hi, _ = paired_bootstrap(hb, h, cl)
            r, _ = rollout(m, room, blend=0.7)
            hs.append(100 * h.mean()); rs.append(r); ds.append(d)
            print(f'   {p.split("/")[-1]:22s} nf={m.nf:3d}  held-out {hs[-1]:6.2f} '
                  f'({d:+5.2f} [{lo:+5.2f},{hi:+5.2f}])   room {r:6.2f}')
        hs, rs = np.array(hs), np.array(rs)
        sd = hs.std(ddof=1) if len(hs) > 1 else 0.0
        summary.append((name, hs.mean(), sd, rs.mean(), float(np.mean(ds)), len(hs)))
        print(f'   CONFIG MEAN  held-out {hs.mean():.2f} (sd {sd:.2f}, '
              f'{np.mean(ds):+.2f} vs ir_only)   room {rs.mean():.2f} '
              f'(sd {rs.std(ddof=1) if len(rs) > 1 else 0:.2f})\n')

    if len(summary) > 1:
        print(f'{"configuration":22s} {"held-out":>9s} {"sd":>5s} {"vs ir":>7s} '
              f'{"room":>7s} {"n":>3s}')
        for nm, h, sd, r, d, n in sorted(summary, key=lambda t: -t[1]):
            print(f'{nm:22s} {h:9.2f} {sd:5.2f} {d:+7.2f} {r:7.2f} {n:3d}')
        best, base = summary[0], [s for s in summary if s[0].startswith('vel_p8')]
        if base:
            b = base[0]
            top = max(summary, key=lambda t: t[1])
            gap = top[1] - b[1]
            pooled = float(np.hypot(top[2], b[2])) or 1e-9
            print(f'\nbest is {top[0]}, {gap:+.2f} held-out over vel_p8 '
                  f'(pooled seed sd {pooled:.2f})')
            print('  ' + ('inside seed noise -- treat as a tie'
                          if abs(gap) < pooled else 'exceeds seed spread'))


if __name__ == '__main__':
    main()
