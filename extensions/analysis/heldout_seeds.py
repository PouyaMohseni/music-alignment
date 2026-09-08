"""Does the CONFIG beat ir_only on held-out, or only the lucky checkpoint?

vel_p8 scores 93.42 on room and is the best of five seeds of its own config,
whose mean is 92.20 against ir_only's 91.44. So the shipped checkpoint sits at
the top of its own draw distribution, and the +2.32 held-out win was measured
on that same checkpoint. A piece-clustered bootstrap samples PIECES; it never
sampled seeds, so its interval is conditional on the draw and understates how
uncertain the configuration is.

This evaluates every seed on the held-out split, each paired against ir_only
over the same onsets. Two things then separate:

  per-seed CIs   whether a typical draw of this config beats ir_only at all
  seed spread    how much of the headline +2.32 was the draw rather than the
                 config

The seeds were trained on FEATK=128 dumps and the held-out dumps are also
FEATK=128, so this is the matched supply for them -- no mismatch inflating the
spread the way candf256 did.
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
from extensions.analysis.heldout_compare import (paired_bootstrap, piece_of,
                                                 rollout_hits)
from extensions.heads.cand_scorer import load as load_ckpt

M = '/scratch/pmohseni/omr/scorer'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='12')
    a = ap.parse_args()
    pages = load_with_feat(f'/scratch/pmohseni/omr/candhv/valid_snr{a.tier}.npz')

    hb, pg = rollout_hits(load_ckpt(f'{M}/ir_only.pt')[0], pages)
    cl = piece_of(pg)
    print(f'held-out snr{a.tier}: {len(hb)} onsets, {len(np.unique(cl))} pieces')
    print(f'ir_only baseline {100.0 * hb.mean():.2f}\n')

    cands = [('vel_p8(seed0)', f'{M}/grid/vel_p8.pt')] + [
        (p.split('/')[-1][:-3], p)
        for p in sorted(glob.glob(f'{M}/seeds/vel_p8_s*.pt'))]
    print(f'{"model":14s} {"pct":>7s} {"delta":>7s} {"95% CI":>18s} {"p":>8s}')
    accs = []
    for n, p in cands:
        h, _ = rollout_hits(load_ckpt(p)[0], pages)
        m, lo, hi, pv = paired_bootstrap(hb, h, cl)
        accs.append(100.0 * h.mean())
        tag = '  RESOLVED' if lo > 0 or hi < 0 else '  n.s.'
        print(f'{n:14s} {accs[-1]:7.2f} {m:+7.2f} [{lo:+6.2f},{hi:+6.2f}] '
              f'{pv:8.4f}{tag}', flush=True)

    v = np.array(accs)
    print(f'\nacross {len(v)} seeds: mean {v.mean():.2f}  sd {v.std(ddof=1):.2f}  '
          f'range {v.max() - v.min():.2f}')
    print(f'config mean beats ir_only by {v.mean() - 100.0 * hb.mean():+.2f}; '
          f'the shipped draw by {v[0] - 100.0 * hb.mean():+.2f}')
    print('\nThe defensible number is the CONFIG mean, not the best draw.')


if __name__ == '__main__':
    main()
