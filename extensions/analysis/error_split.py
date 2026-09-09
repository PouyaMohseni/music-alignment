"""Split the 6.15 points: how much is error propagation, how much is ranking?

The real-time ORACLE ceiling is 99.57 on room and we ship 93.42. But the oracle
knows which candidate is right; the gap is not a target, it is an upper bound
proving the answer is present and reachable online. What matters is where the
gap lives, because the two halves need completely different work:

  A  deployed      the decoder conditions on its OWN previous position   93.42
  B  teacher-forced   ... on the TRUE previous position, every frame        ?
  C  real-time oracle ... and it also knows the answer                   99.57

  B - A  is ERROR PROPAGATION. The ranking is this good when handed a correct
         history and this much is lost to conditioning on its own mistakes.
         Committee tracking, recovery and DAgger all target exactly this.
  C - B  is RANKING QUALITY. Even given a perfect history the scorer still
         misses this much, and only better features or a better model close it.

A third arm prices propagation directly: reset the history to ground truth
every k onsets and watch the curve run from A to B.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.offline_decode import TH, prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt

REF, FWD, SIG, JUMP = 5.0, 6.0, 18.0, -6.0


def oracle_xy(c, t_gt):
    j = int(np.argmin(np.abs(c[:, 5] - t_gt)))
    return float(c[j, 0]), float(c[j, 1])


def run(model, pages, blend=0.7, lam=1.0, topk=256, reset_every=0,
        teacher=False):
    """reset_every=k snaps the history back to truth every k onsets.
    teacher=True is the k=1 limit: a true history at every frame."""
    hit = tot = 0
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
        feats = p.get('feat')
        since = 0
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
                    hand = lo + lam * prior_logp(cs[:, 0] - x_prev, FWD * k,
                                                 SIG, JUMP)
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH

            since += 1
            snap = teacher or (reset_every and since >= reset_every)
            if snap:
                # hand the next step a TRUE history instead of this one's choice
                xo, yo = oracle_xy(cs, p['t_gt'][i])
                nx, ny = xo, yo
                since = 0
            else:
                nx, ny = float(cs[j, 0]), float(cs[j, 1])
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = nx, ny, fr
    return 100.0 * hit / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--dump', default='/scratch/pmohseni/omr/candf256/room.npz')
    a = ap.parse_args()
    m = load_ckpt(a.ckpt)[0]
    pages = load_with_feat(a.dump)

    dep, n = run(m, pages)
    tea, _ = run(m, pages, teacher=True)
    print(f'{n} onsets\n')
    print(f'  A  deployed (own history)          {dep:6.2f}')
    print(f'  B  teacher-forced (true history)   {tea:6.2f}')
    print(f'  C  real-time oracle ceiling         99.57')
    print(f'\n  B - A  error propagation           {tea - dep:6.2f}')
    print(f'  C - B  ranking quality             {99.57 - tea:6.2f}')

    print(f'\n=== how fast does error accumulate? (snap to truth every k) ===')
    print(f'{"k":>4s} {"acc":>7s}')
    for k in (1, 2, 3, 5, 10, 20, 50):
        acc, _ = run(m, pages, reset_every=k)
        print(f'{k:4d} {acc:7.2f}')
    print(f'{"never":>4s} {dep:7.2f}')


if __name__ == '__main__':
    main()
