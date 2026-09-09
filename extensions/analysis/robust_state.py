"""Make the STATE outlier-resistant, instead of detecting when it is wrong.

The decomposition says where the headroom is: teacher forcing scores 98.07
against a deployed 93.42 and a 99.57 oracle, so 4.65 of the 6.15 points are
ERROR PROPAGATION and only 1.50 is ranking. The decoder conditions on its own
last choice, and one wrong choice poisons every step after it.

Everything tried against that has needed to KNOW it was lost -- gated recovery,
soft recovery, DAgger, committee arbitration -- and all four failed on the same
wall: the lostness signal separates at AUC 0.805 but only 0.33 precision, which
is unusable as a gate, a weight, or an arbitrator.

This needs no such decision. Instead of conditioning on the single last
position, fit the recent trajectory and use the fitted value. A line through
five recent positions barely moves when one of them is wrong, so a single bad
step is attenuated rather than inherited -- no detection, no jump, no
arbitration, nothing to get right.

  last    the shipped behaviour, and the control: x_prev is the last choice
  ls      least squares over the last k (frame, x) pairs, read back at the
          previous frame. Smooths, but one gross outlier still drags the fit
  theil   Theil-Sen: the MEDIAN of pairwise slopes. A single wild step cannot
          move a median, which is the whole point
  shrink  a blend between the last choice and where the earlier steps predict
          it should have been, with the weight swept

The fit is read back AT THE PREVIOUS FRAME rather than extrapolated to now, so
every downstream feature and the prior's mu keep their existing meaning and
mode='last' is bit-identical to the shipped decoder.
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

REF, FWD, SIG, JUMP = 5.0, 6.0, 18.0, -6.0


def robust_prev(hist, mode, k, shrink):
    """hist: list of (frame, x) the decoder actually chose, oldest first."""
    if not hist:
        return None
    f_last, x_last = hist[-1]
    if mode == 'last' or len(hist) < 3:
        return x_last
    h = hist[-k:]
    fs = np.array([a for a, _ in h], float)
    xs = np.array([b for _, b in h], float)
    if np.ptp(fs) <= 0:
        return x_last
    if mode == 'ls':
        b, a = np.polyfit(fs, xs, 1)
        return float(a + b * f_last)
    if mode == 'theil':
        sl = [(xs[j] - xs[i]) / (fs[j] - fs[i])
              for i, j in itertools.combinations(range(len(h)), 2)
              if fs[j] != fs[i]]
        if not sl:
            return x_last
        b = float(np.median(sl))
        a = float(np.median(xs - b * fs))
        return a + b * f_last
    if mode == 'shrink':
        # where the EARLIER steps say we should be, ignoring the last one
        if len(h) < 4:
            return x_last
        b, a = np.polyfit(fs[:-1], xs[:-1], 1)
        pred = float(a + b * f_last)
        return (1.0 - shrink) * x_last + shrink * pred
    return x_last


def run(model, pages, blend=0.7, lam=1.0, topk=256, mode='last', k=5,
        shrink=0.5):
    hit = tot = 0
    for p in pages:
        hist, y_prev = [], None
        x_prev = x_prev2 = f_prev = f_prev2 = None
        feats = p.get('feat')
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
            f = build(cs, p['bar'][i], p['sys'][i], x_prev, y_prev, dfr,
                      ntot=int(p['ntot'][i]), use_abs_obj=model.use_abs_obj,
                      x_prev2=x_prev2, dframes_prev=dfp)[:, :model.nf]
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
                if x_prev is None:
                    hand = lo
                else:
                    kk = np.clip((dfr or REF) / REF, 0.2, 8.0)
                    hand = lo + lam * prior_logp(cs[:, 0] - x_prev, FWD * kk,
                                                 SIG, JUMP)
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            hist.append((fr, float(cs[j, 0])))
            # the NEXT step conditions on a fitted history, not this one choice
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev = robust_prev(hist, mode, k, shrink)
            y_prev, f_prev = float(cs[j, 1]), fr
    return 100.0 * hit / max(tot, 1), tot


_G = {}


def _init(ckpt, dumps):
    _G['m'] = load_ckpt(ckpt)[0]
    for a_, b_ in dumps.items():
        _G[a_] = load_with_feat(b_)


def _run(job):
    which, mode, k, sh = job
    acc, _ = run(_G['m'], _G[which], mode=mode, k=k, shrink=sh)
    return which, mode, k, sh, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    dumps = {'do': '/scratch/pmohseni/omr/candf/do.npz',
             'room': '/scratch/pmohseni/omr/candf256/room.npz'}
    jobs = [('do', 'last', 5, 0.0)]
    for mode in ('ls', 'theil'):
        jobs += [('do', mode, k, 0.0) for k in (3, 5, 8, 12)]
    jobs += [('do', 'shrink', k, sh) for k in (4, 6, 10)
             for sh in (0.25, 0.5, 0.75)]
    with Pool(a.procs, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        res = pool.map(_run, jobs)
    ctrl = [r for r in res if r[1] == 'last'][0]
    print(f'control (x_prev = last choice) on do: {ctrl[4]:.2f}  '
          f'(must be 95.01)\n')
    print(f'{"mode":>8s} {"k":>3s} {"shrink":>7s} {"do":>7s} {"delta":>7s}')
    for _, m_, k_, s_, acc in sorted([r for r in res if r[1] != 'last'],
                                     key=lambda r: -r[4]):
        print(f'{m_:>8s} {k_:3d} {s_:7.2f} {acc:7.2f} {acc - ctrl[4]:+7.2f}')
    best = max([r for r in res if r[1] != 'last'], key=lambda r: r[4])
    print(f'\n`do` picks {best[1]} k={best[2]} shrink={best[3]}')
    if best[4] <= ctrl[4]:
        print('  nothing beats the control on do; room would only confirm it')
    print('=== the one look at room ===')
    with Pool(2, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        rr = pool.map(_run, [('room', 'last', 5, 0.0),
                             ('room', best[1], best[2], best[3])])
    for _, m_, k_, s_, acc in rr:
        print(f'  {m_:>8s} k={k_:2d} shrink={s_:.2f}   room {acc:6.2f}')


if __name__ == '__main__':
    main()
