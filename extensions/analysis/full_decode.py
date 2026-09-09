"""Every decoder lever in one place, searched jointly rather than one at a time.

The levers, and why each is here:

  sigma_x     the ceiling analysis says a real-time decoder limited to <=100 px
              of forward travel tops out at 96.63 and an unbounded one at
              99.57. fwd_px=6 / sigma_px=18 puts us in the 100 px regime, so
              the reach is a measured cap and not a taste parameter.
  fwd_ratio   widen FORWARD only. Ground truth steps backwards on 0.73% of
              onsets, the decoder on 4.50%: the asymmetry is in the data.
  jump        what a far candidate pays once the tail takes over.
  lam_y       THE PRIOR HAS NO VERTICAL TERM AT ALL. It scores unrolled-x
              displacement and nothing else, yet 25.4% of errors are "correct
              horizontally, jumped to a neighbouring system" -- the right place
              within a staff, on the wrong staff. In unrolled coordinates that
              is a large x move, so the x prior should already reject it, and
              evidently the tail is cheap enough that objectness wins. A
              y-continuity term prices the staff change directly.
  sys_slack   keep candidates inside the predicted system box (91.7% accurate).
              Same target as lam_y by a different route: a hard filter rather
              than a soft cost, so they are swept together and may substitute
              for one another.

COORDINATE DESCENT, not a grid. Five knobs at four values each is 1024 cells
and most are nonsense; two passes of one-at-a-time refinement over ~40 cells
finds the same optimum in a fortieth of the time, and the trace shows which
lever actually paid.

Tuned on `do`, room read ONCE at the end. Every constant in the shipped decoder
was originally swept on room, which is why none of them was ever defensible.
"""
from __future__ import annotations

import argparse
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
SHIPPED = dict(sigma_x=18.0, fwd_ratio=1.0, jump=-6.0, lam_y=0.0,
               sigma_y=40.0, sys_slack=0.0)


def run(model, pages, blend=0.7, lam=1.0, topk=256, sigma_x=18.0,
        fwd_ratio=1.0, jump=-6.0, lam_y=0.0, sigma_y=40.0, sys_slack=0.0):
    hit = tot = 0
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
        feats = p.get('feat')
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            tot += 1
            keep = np.ones(cs.shape[0], bool)
            if sys_slack > 0:
                sb = p['sys'][i]
                if float(sb[4]) > 0:
                    lo_, hi_ = sb[1] - sb[3] / 2 - sys_slack, sb[1] + sb[3] / 2 + sys_slack
                    k2 = (cs[:, 1] >= lo_) & (cs[:, 1] <= hi_)
                    # never manufacture a no-detection frame
                    if k2.any():
                        keep = k2
            idxs = np.flatnonzero(keep)
            sub = cs[idxs]
            fr = int(p['frame'][i])
            dfr = fr - f_prev if f_prev is not None and fr > f_prev else None
            dfp = (f_prev - f_prev2
                   if f_prev is not None and f_prev2 is not None and f_prev > f_prev2
                   else None)
            f = build(sub, p['bar'][i], p['sys'][i], x_prev, y_prev, dfr,
                      ntot=int(p['ntot'][i]), use_abs_obj=model.use_abs_obj,
                      x_prev2=x_prev2, dframes_prev=dfp)[:, :model.nf]
            ff = None
            if model.fenc is not None and feats is not None:
                fv = feats[i].astype(np.float32)
                if fv.shape[0] < cs.shape[0]:
                    fv = np.vstack([fv, np.zeros((cs.shape[0] - fv.shape[0],
                                                  model.featdim), np.float32)])
                ff = torch.from_numpy(fv[idxs]).unsqueeze(0)
            with torch.no_grad():
                s = model(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()
            if blend < 1.0:
                lo = np.log(np.clip(sub[:, 4], 1e-8, None))
                if x_prev is None:
                    hand = lo
                else:
                    k = np.clip((dfr or REF) / REF, 0.2, 8.0)
                    d = sub[:, 0] - x_prev
                    mu = FWD * k
                    sg = np.where(d >= mu, sigma_x * fwd_ratio, sigma_x)
                    hand = lo + lam * np.maximum(-0.5 * ((d - mu) / sg) ** 2, jump)
                    if lam_y > 0 and y_prev is not None:
                        dy = sub[:, 1] - y_prev
                        hand = hand + lam_y * np.maximum(
                            -0.5 * (dy / sigma_y) ** 2, jump)
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hit += abs(float(sub[j, 5]) - float(p['t_gt'][i])) <= TH
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = float(sub[j, 0]), float(sub[j, 1]), fr
    return 100.0 * hit / max(tot, 1), tot


GRID = dict(
    sigma_x=[18.0, 30.0, 60.0, 120.0],
    fwd_ratio=[1.0, 2.0, 4.0],
    jump=[-6.0, -4.0, -8.0, -3.0],
    lam_y=[0.0, 0.5, 1.0, 2.0],
    sigma_y=[20.0, 40.0, 80.0],
    sys_slack=[0.0, 10.0, 30.0, 80.0],
)
_G = {}


def _init(ckpt, dumps):
    _G['m'] = load_ckpt(ckpt)[0]
    for k, v in dumps.items():
        _G[k] = load_with_feat(v)


def _run(job):
    which, cfg = job
    acc, _ = run(_G['m'], _G[which], **cfg)
    return which, cfg, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--passes', type=int, default=2)
    a = ap.parse_args()
    dumps = {'do': '/scratch/pmohseni/omr/candf/do.npz',
             'room': '/scratch/pmohseni/omr/candf256/room.npz'}
    pool = Pool(a.procs, initializer=_init, initargs=(a.ckpt, dumps))

    cur = dict(SHIPPED)
    base, _ = pool.map(_run, [('do', dict(cur))])[0][2], None
    print(f'shipped decoder on do: {base:.2f}   (must be 95.01)\n')
    for ps in range(a.passes):
        for knob, vals in GRID.items():
            jobs = []
            for v in vals:
                cfg = dict(cur); cfg[knob] = v
                jobs.append(('do', cfg))
            res = pool.map(_run, jobs)
            best = max(res, key=lambda r: r[2])
            mark = '' if best[1][knob] == cur[knob] else '  <- moved'
            print(f'pass {ps + 1}  {knob:10s} '
                  + ' '.join(f'{v:g}:{r[2]:.2f}' for v, r in zip(vals, res))
                  + f'   keep {best[1][knob]:g}{mark}')
            cur[knob] = best[1][knob]
    fin = pool.map(_run, [('do', dict(cur))])[0][2]
    print(f'\ntuned on do: {fin:.2f}  ({fin - base:+.2f})')
    print(f'  {cur}')
    print('\n=== the one look at room ===')
    rr = pool.map(_run, [('room', dict(SHIPPED)), ('room', dict(cur))])
    for _, cfg, acc in rr:
        tag = 'shipped' if cfg == SHIPPED else 'tuned  '
        print(f'  {tag}  room {acc:6.2f}')
    pool.close()


if __name__ == '__main__':
    main()
