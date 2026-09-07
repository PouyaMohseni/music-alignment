"""The whole room ladder from one dump, with paired CIs on every step.

The headline is a chain -- argmax -> hand decoder -> selector -> vel_p8 -- and
so far only the top step has a confidence interval. A paper needs the interval
on each step, and on the end-to-end claim, computed on the SAME onsets so the
comparisons are paired rather than four independent measurements.

Every arm is decoded from the identical dump, so the only thing that varies is
the decision rule:

  argmax   pick the highest objectness. This is what cyolo_sb does and it must
           land near 80.0.
  hand     blend=0.0, the hand-tuned prior. Must land on 86.5.
  ir_only  blend=0.7, the shipped selector. Must land on 91.4.
  vel_p8   blend=0.7 with velocity features.

The first three are gates: they are known numbers, and if they do not come back
the harness is not being reproduced and nothing below is worth reading.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import (paired_bootstrap, piece_of,
                                                 rollout_hits)
from extensions.analysis.offline_decode import TH, prior_logp
from extensions.heads.cand_scorer import load as load_ckpt


def rollout_argmax(pages, topk=256):
    """cyolo_sb's own rule: highest objectness, no history at all."""
    hits, page_of = [], []
    for p in pages:
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            j = int(np.argmax(cs[:, 4]))
            hits.append(abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH)
            page_of.append(p['name'])
    return np.array(hits, bool), np.array(page_of)


def rollout_hand(pages, lam=1.0, fwd=6.0, sigma=18.0, jump=-6.0, ref=5.0,
                 mu_pow=1.0, topk=256):
    hits, page_of = [], []
    for p in pages:
        x_prev = f_prev = None
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            fr = int(p['frame'][i])
            dfr = fr - f_prev if f_prev is not None and fr > f_prev else None
            lo = np.log(np.clip(cs[:, 4], 1e-8, None))
            if x_prev is None:
                s = lo
            else:
                k = np.clip((dfr or ref) / ref, 0.2, 8.0) ** mu_pow
                s = lo + lam * prior_logp(cs[:, 0] - x_prev, fwd * k, sigma, jump)
            j = int(np.argmax(s))
            hits.append(abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH)
            page_of.append(p['name'])
            x_prev, f_prev = float(cs[j, 0]), fr
    return np.array(hits, bool), np.array(page_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/scratch/pmohseni/omr/candf/room.npz')
    a = ap.parse_args()
    pages = load_with_feat(a.dump)

    arms = {}
    arms['argmax'], pg = rollout_argmax(pages)
    arms['hand'], _ = rollout_hand(pages)
    arms['ir_only'], _ = rollout_hits(load_ckpt('/scratch/pmohseni/omr/scorer/ir_only.pt')[0], pages)
    arms['vel_p8'], _ = rollout_hits(load_ckpt('/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')[0], pages)
    cl = piece_of(pg)

    print(f'{len(arms["argmax"])} onsets, {len(np.unique(pg))} pages, '
          f'{len(np.unique(cl))} pieces\n')
    gates = {'argmax': 80.0, 'hand': 86.5, 'ir_only': 91.4}
    print(f'{"arm":>9s} {"pct@0.5s":>9s}   gate')
    for k, h in arms.items():
        got = 100.0 * h.mean()
        g = gates.get(k)
        note = '' if g is None else (
            f'  expect {g}  ' + ('OK' if abs(got - g) < 1.0 else 'MISMATCH'))
        print(f'{k:>9s} {got:9.2f}{note}')

    print(f'\n{"comparison":>22s} {"delta":>7s} {"95% CI":>18s} {"p":>8s}')
    steps = [('argmax', 'hand'), ('hand', 'ir_only'), ('ir_only', 'vel_p8'),
             ('argmax', 'ir_only'), ('argmax', 'vel_p8')]
    for b, c in steps:
        m, lo, hi, p = paired_bootstrap(arms[b], arms[c], cl)
        tag = '  RESOLVED' if lo > 0 or hi < 0 else '  n.s.'
        print(f'{b + " -> " + c:>22s} {m:+7.2f} [{lo:+6.2f},{hi:+6.2f}] '
              f'{p:8.4f}{tag}')


if __name__ == '__main__':
    main()
