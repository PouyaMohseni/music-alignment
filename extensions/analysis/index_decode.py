"""Greedy decode with the transition prior in NOTE-INDEX space.

Identical to the shipped rule except the displacement fed to the prior is a
difference of note indices rather than pixels, so fwd/sigma are in noteheads.
`use_index=False` must reproduce the pixel decoder exactly, which is the gate.
"""
from __future__ import annotations

import numpy as np

from extensions.analysis.note_index import build_bins, to_index
from extensions.analysis.offline_decode import TH, prior_logp


def score(pages, use_index=True, bin_px=12.0, lam=1.0, fwd=1.0, sigma=3.0,
          jump=-6.0, ref=5.0, mu_pow=1.0, topk=256, back=None):
    bins = build_bins(pages, bin_px=bin_px) if use_index else None
    hit = tot = 0
    for p in pages:
        ctr = bins[p['name']] if use_index else None
        x_prev = f_prev = None
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            tot += 1
            pos = to_index(cs[:, 0], ctr) if use_index else cs[:, 0]
            lo = np.log(np.clip(cs[:, 4], 1e-8, None))
            fr = int(p['frame'][i])
            if x_prev is None:
                s = lo
            else:
                df = fr - f_prev if f_prev is not None and fr > f_prev else ref
                k = np.clip(df / ref, 0.2, 8.0) ** mu_pow
                d = pos - x_prev
                pr = prior_logp(d, fwd * k, sigma, jump)
                if back is not None:
                    pr = np.maximum(-0.5 * ((d - fwd * k) / sigma) ** 2,
                                    np.where(d < 0, back, jump))
                s = lo + lam * pr
            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            x_prev, f_prev = float(pos[j]), fr
    return 100.0 * hit / max(tot, 1), tot
