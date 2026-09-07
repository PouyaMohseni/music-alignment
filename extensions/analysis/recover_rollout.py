"""Gated recovery: re-localize from scratch when the decoder stops believing itself.

WHY GATED, AND WHY THIS GATE
----------------------------
Re-anchoring and wider beams both LOST on room (82.8-85.7 and 84.7-83.8 against
86.5) because they fired on every frame, paying a cost everywhere to fix
something that happens in ~18 places. The episode analysis says 71% of lost
onsets sit in contiguous runs of >=5, and the signal analysis says those runs
are visible: `obj_chosen` separates in-episode from fine at AUC 0.805.

The mechanism behind that number is what makes it trustworthy rather than a
lucky correlation. A lost tracker stays locally consistent in POSITION -- the
transition residual is worth only 0.558 -- because each step is still a small
plausible displacement from a wrong x_prev. What it cannot fake is EVIDENCE:
pinned near a place where no note exists, it must pick a box the detector does
not believe in. Low objectness on the chosen candidate is the fingerprint of
being somewhere there is no music.

THE ACTION
----------
When the gate fires, rebuild the features with x_prev=None and score without the
transition prior. That is exactly the cold-start condition at the top of a page,
which the model is already trained for, so recovery reuses learned behaviour
instead of introducing a new rule. History is then reset so the next step
tracks from the recovered position.

TUNED ON `do`, REPORTED ON room. Picking thresholds on room would make the
result unquotable, and that proxy already picked vel_p8 correctly.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from multiprocessing import Pool

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.offline_decode import TH, prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt


def rollout(model, pages, blend=0.7, lam=1.0, fwd=6.0, sigma=18.0, jump=-6.0,
            ref=5.0, mu_pow=1.0, topk=256, tau=0.0, win=3, cooldown=5):
    """tau=0 disables the gate, reproducing the shipped rollout exactly."""
    hit = tot = fired = 0
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
        feats, hist, cool = p.get('feat'), [], 0
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            tot += 1
            fr = int(p['frame'][i])
            dfr = fr - f_prev if f_prev is not None and fr > f_prev else None
            dfp = (f_prev - f_prev2
                   if f_prev is not None and f_prev2 is not None and f_prev > f_prev2
                   else None)

            # GATE: sustained low belief in what we have been choosing
            lost = (tau > 0 and x_prev is not None and cool <= 0
                    and len(hist) >= win and float(np.mean(hist[-win:])) < tau)
            if lost:
                fired += 1
                xp, yp, xp2, dd, dp = None, None, None, None, None
            else:
                xp, yp, xp2, dd, dp = x_prev, y_prev, x_prev2, dfr, dfp

            f = build(cs, p['bar'][i], p['sys'][i], xp, yp, dd,
                      ntot=int(p['ntot'][i]), use_abs_obj=model.use_abs_obj,
                      x_prev2=xp2, dframes_prev=dp)[:, :model.nf]
            ff = None
            if model.fenc is not None and feats is not None:
                fv = feats[i].astype(np.float32)
                if fv.shape[0] < cs.shape[0]:
                    fv = np.vstack([fv, np.zeros((cs.shape[0] - fv.shape[0],
                                                  model.featdim), np.float32)])
                ff = torch.from_numpy(fv[:cs.shape[0]]).unsqueeze(0)
            with torch.no_grad():
                s = model(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()

            if blend < 1.0:
                lo = np.log(np.clip(cs[:, 4], 1e-8, None))
                if xp is None:
                    hand = lo                      # cold start: evidence only
                else:
                    k = np.clip((dd or ref) / ref, 0.2, 8.0) ** mu_pow
                    hand = lo + lam * prior_logp(cs[:, 0] - xp, fwd * k, sigma, jump)
                s = blend * s + (1.0 - blend) * hand

            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            hist.append(float(cs[j, 4]))
            cool = cooldown if lost else cool - 1
            if lost:
                x_prev2, f_prev2 = None, None
            else:
                x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = float(cs[j, 0]), float(cs[j, 1]), fr
    return 100.0 * hit / max(tot, 1), tot, fired


_G = {}


def _init(ckpt, dumps):
    _G['model'] = load_ckpt(ckpt)[0]
    for nm, d in dumps.items():
        _G[nm] = load_with_feat(d)


def _run(job):
    which, tau, win, cd = job
    acc, n, fired = rollout(_G['model'], _G[which], tau=tau, win=win, cooldown=cd)
    return which, tau, win, cd, acc, fired, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--do', default='/scratch/pmohseni/omr/candf/do.npz')
    ap.add_argument('--room', default='/scratch/pmohseni/omr/candf/room.npz')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()

    dumps = {'do': a.do, 'room': a.room}
    taus = [0.03, 0.05, 0.08, 0.12, 0.20, 0.30]
    wins, cds = [1, 3, 5], [3, 10]
    jobs = [('do', 0.0, 1, 3)] + [('do', t, w, c)
                                  for t, w, c in itertools.product(taus, wins, cds)]

    with Pool(a.procs, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        res = pool.map(_run, jobs)

    ctrl = [r for r in res if r[1] == 0.0][0]
    print(f'CONTROL (gate off) on do: {ctrl[4]:.2f}  ({ctrl[6]} onsets)\n')
    print(f'{"tau":>6s} {"win":>4s} {"cd":>4s} {"do":>7s} {"d_do":>7s} {"fired":>6s}')
    swept = sorted([r for r in res if r[1] > 0], key=lambda r: -r[4])
    for _, t, w, c, acc, fired, _ in swept:
        print(f'{t:6.2f} {w:4d} {c:4d} {acc:7.2f} {acc - ctrl[4]:+7.2f} {fired:6d}')

    best = swept[0]
    print(f'\n`do` picks tau={best[1]} win={best[2]} cd={best[3]} '
          f'-> do {best[4]:.2f} ({best[4] - ctrl[4]:+.2f})')

    print('\n=== the one look at room ===')
    with Pool(2, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        rr = pool.map(_run, [('room', 0.0, 1, 3),
                             ('room', best[1], best[2], best[3])])
    rc = [r for r in rr if r[1] == 0.0][0]
    rb = [r for r in rr if r[1] > 0][0]
    print(f'  room control      {rc[4]:.2f}')
    print(f'  room + recovery   {rb[4]:.2f}  ({rb[4] - rc[4]:+.2f}, '
          f'gate fired {rb[5]} times in {rb[6]} onsets)')


if __name__ == '__main__':
    main()
