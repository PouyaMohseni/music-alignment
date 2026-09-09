"""Widen the prior's forward reach, which is what actually caps us.

The ceiling analysis found that a real-time decoder limited to <=100 px of
forward travel per onset tops out at 96.63, at <=200 px at 98.41, and with
unbounded reach at 99.57. Our prior is fwd_px=6 with sigma_px=18, so it lives
in the 100 px regime -- we are near the ceiling OUR PRIOR allows (96.6) rather
than the real one (99.6), and we ship 93.42.

So the constants are swept again, but for a reason this time and against the
right target. Three knobs:

  sigma      how far a candidate may sit from the expected step before the
             heavy tail takes over
  fwd_ratio  asymmetry: forward moves get sigma * fwd_ratio, backward moves
             get sigma. Ground truth steps backwards on 0.73% of onsets and
             the decoder does on 4.50%, so widening forward WITHOUT widening
             backward is the shape the data asks for
  jump       the floor a far-away candidate pays once the tail takes over

The earlier sweep of these was tuned on room, which makes its optimum
undefendable. This one is tuned on `do` and reads room exactly once.
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
from extensions.analysis.offline_decode import TH
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt

REF, FWD = 5.0, 6.0


def asym_prior(d, mu, sigma, fwd_ratio, jump):
    """Gaussian core with a heavy-tail floor, wider ahead than behind."""
    sg = np.where(d >= mu, sigma * fwd_ratio, sigma)
    return np.maximum(-0.5 * ((d - mu) / sg) ** 2, jump)


def run(model, pages, blend=0.7, lam=1.0, topk=256, sigma=18.0,
        fwd_ratio=1.0, jump=-6.0, fwd=FWD):
    hit = tot = 0
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
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
                    k = np.clip((dfr or REF) / REF, 0.2, 8.0)
                    hand = lo + lam * asym_prior(cs[:, 0] - x_prev, fwd * k,
                                                 sigma, fwd_ratio, jump)
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = float(cs[j, 0]), float(cs[j, 1]), fr
    return 100.0 * hit / max(tot, 1), tot


_G = {}


def _init(ckpt, dumps):
    _G['m'] = load_ckpt(ckpt)[0]
    for k, v in dumps.items():
        _G[k] = load_with_feat(v)


def _run(job):
    which, sg, fr_, jp = job
    acc, _ = run(_G['m'], _G[which], sigma=sg, fwd_ratio=fr_, jump=jp)
    return which, sg, fr_, jp, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    dumps = {'do': '/scratch/pmohseni/omr/candf/do.npz',
             'room': '/scratch/pmohseni/omr/candf256/room.npz'}
    grid = list(itertools.product((18.0, 30.0, 60.0, 120.0),
                                  (1.0, 2.0, 4.0), (-6.0, -4.0, -8.0)))
    jobs = [('do', 18.0, 1.0, -6.0)] + [('do', s, f, j) for s, f, j in grid]
    with Pool(a.procs, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        res = pool.map(_run, jobs)
    ctrl = [r for r in res if (r[1], r[2], r[3]) == (18.0, 1.0, -6.0)][0]
    print(f'control (shipped prior) on do: {ctrl[4]:.2f}  '
          f'(must be 95.01)\n')
    print(f'{"sigma":>6s} {"fwd_x":>6s} {"jump":>5s} {"do":>7s} {"delta":>7s}')
    top = sorted(res, key=lambda r: -r[4])[:12]
    for _, s_, f_, j_, acc in top:
        print(f'{s_:6.0f} {f_:6.1f} {j_:5.1f} {acc:7.2f} {acc - ctrl[4]:+7.2f}')
    best = top[0]
    print(f'\n`do` picks sigma={best[1]} fwd_ratio={best[2]} jump={best[3]}')
    print('=== the one look at room ===')
    with Pool(2, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        rr = pool.map(_run, [('room', 18.0, 1.0, -6.0),
                             ('room', best[1], best[2], best[3])])
    for _, s_, f_, j_, acc in rr:
        tag = 'shipped prior' if (s_, f_, j_) == (18.0, 1.0, -6.0) else 'widened'
        print(f'  {tag:16s} sigma={s_:5.0f} fwd_x={f_:3.1f} jump={j_:4.1f}  '
              f'room {acc:6.2f}')


if __name__ == '__main__':
    main()
