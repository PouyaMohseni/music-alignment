"""Soft recovery: weight the cold-start score by belief instead of jumping.

The hard gate failed for a specific, measured reason. Precision never exceeds
0.33 at any operating point, so most fires land on healthy tracking, and a fire
was all-or-nothing: it discarded history and re-localized. At the mildest
setting that recovered 46 lost onsets while destroying 93 good locks, a net
-47 before the cascades that made the observed loss -2.0.

The information and the cost are separable, and only one of them was tested.
This changes the cost: instead of switching rules, mix them by belief,

    s = (1 - b) * s_tracked + b * s_coldstart,   b in [0, 1]

so a false positive on a good lock applies a small wrong pressure that the
tracked term outvotes, rather than throwing the lock away. If the binding
constraint was the cost asymmetry, this recovers some of the 218 in-episode
onsets. If it was the information -- a signal right one time in three cannot
steer anything -- this fails too, and the lock-loss lever is closed for good.

b = 0 everywhere reproduces the shipped rollout exactly.
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


def _score(model, cs, bar, sys_, xp, yp, dd, xp2, dp, ntot, blend, lam,
           fwd, sigma, jump, ref, mu_pow, ff):
    f = build(cs, bar, sys_, xp, yp, dd, ntot=ntot,
              use_abs_obj=model.use_abs_obj, x_prev2=xp2,
              dframes_prev=dp)[:, :model.nf]
    with torch.no_grad():
        s = model(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()
    if blend >= 1.0:
        return s
    lo = np.log(np.clip(cs[:, 4], 1e-8, None))
    if xp is None:
        hand = lo
    else:
        k = np.clip((dd or ref) / ref, 0.2, 8.0) ** mu_pow
        hand = lo + lam * prior_logp(cs[:, 0] - xp, fwd * k, sigma, jump)
    return blend * s + (1.0 - blend) * hand


def rollout(model, pages, blend=0.7, lam=1.0, fwd=6.0, sigma=18.0, jump=-6.0,
            ref=5.0, mu_pow=1.0, topk=256, tau=0.0, win=3, gain=1.0):
    hit = tot = 0
    bsum = 0.0
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
        feats, hist = p.get('feat'), []
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
            ff = None
            if model.fenc is not None and feats is not None:
                fv = feats[i].astype(np.float32)
                if fv.shape[0] < cs.shape[0]:
                    fv = np.vstack([fv, np.zeros((cs.shape[0] - fv.shape[0],
                                                  model.featdim), np.float32)])
                ff = torch.from_numpy(fv[:cs.shape[0]]).unsqueeze(0)

            args = (blend, lam, fwd, sigma, jump, ref, mu_pow, ff)
            s = _score(model, cs, p['bar'][i], p['sys'][i], x_prev, y_prev, dfr,
                       x_prev2, dfp, int(p['ntot'][i]), *args)

            # continuous belief that we are somewhere with no music
            b = 0.0
            if tau > 0 and x_prev is not None and len(hist) >= win:
                roll = float(np.mean(hist[-win:]))
                b = float(np.clip(gain * (tau - roll) / max(tau, 1e-6), 0.0, 1.0))
            if b > 0.01:
                s_cold = _score(model, cs, p['bar'][i], p['sys'][i], None, None,
                                None, None, None, int(p['ntot'][i]), *args)
                s = (1.0 - b) * s + b * s_cold
            bsum += b

            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            hist.append(float(cs[j, 4]))
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = float(cs[j, 0]), float(cs[j, 1]), fr
    return 100.0 * hit / max(tot, 1), tot, bsum


_G = {}


def _init(ckpt, dumps):
    _G['model'] = load_ckpt(ckpt)[0]
    for nm, d in dumps.items():
        _G[nm] = load_with_feat(d)


def _run(job):
    which, tau, win, gain = job
    acc, n, bsum = rollout(_G['model'], _G[which], tau=tau, win=win, gain=gain)
    return which, tau, win, gain, acc, bsum, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--do', default='/scratch/pmohseni/omr/candf/do.npz')
    ap.add_argument('--room', default='/scratch/pmohseni/omr/candf/room.npz')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    dumps = {'do': a.do, 'room': a.room}
    jobs = [('do', 0.0, 3, 1.0)] + [
        ('do', t, w, g) for t, w, g in itertools.product(
            [0.05, 0.10, 0.20, 0.30], [3, 5], [0.5, 1.0])]

    with Pool(a.procs, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        res = pool.map(_run, jobs)
    ctrl = [r for r in res if r[1] == 0.0][0]
    print(f'CONTROL (b=0) on do: {ctrl[4]:.2f}\n')
    print(f'{"tau":>6s} {"win":>4s} {"gain":>5s} {"do":>7s} {"d_do":>7s} '
          f'{"mean b":>7s}')
    swept = sorted([r for r in res if r[1] > 0], key=lambda r: -r[4])
    for _, t, w, g, acc, bsum, n in swept:
        print(f'{t:6.2f} {w:4d} {g:5.1f} {acc:7.2f} {acc - ctrl[4]:+7.2f} '
              f'{bsum / n:7.3f}')
    best = swept[0]
    print(f'\n`do` picks tau={best[1]} win={best[2]} gain={best[3]} '
          f'-> {best[4]:.2f} ({best[4] - ctrl[4]:+.2f})')
    if best[4] <= ctrl[4]:
        print('  nothing beats the control on do; room would only confirm a loss')
    print('\n=== the one look at room ===')
    with Pool(2, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        rr = pool.map(_run, [('room', 0.0, 3, 1.0),
                             ('room', best[1], best[2], best[3])])
    rc = [r for r in rr if r[1] == 0.0][0]
    rb = [r for r in rr if r[1] > 0][0]
    print(f'  room control    {rc[4]:.2f}')
    print(f'  room + soft     {rb[4]:.2f}  ({rb[4] - rc[4]:+.2f}, '
          f'mean belief {rb[5] / rb[6]:.3f})')


if __name__ == '__main__':
    main()
