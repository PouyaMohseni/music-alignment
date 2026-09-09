"""Track tempo instead of assuming it -- Antescofo's coupled agent, minimally.

MEASURED MOTIVATION
-------------------
The shipped prior expects fwd_px=6.0 px per REF_FRAMES=5, i.e. 1.2 px/frame,
for every piece. Ground truth says the real figure is 3.79 px/frame, and 24 of
25 pages are more than 2x off the assumption. Pieces span 1.91 to 9.11, a 4.8x
spread; within a piece the IQR is 0.54 of the median, so most of the variation
is BETWEEN pieces and a tracked per-piece value should capture it.

The bias has a signature we already measured and never attributed. A mu that is
too small pulls the decoder toward candidates BEHIND the true position, and the
decoder steps backward on 4.50% of onsets where ground truth does so on 0.73%
-- six times too often. Under-set expected travel predicts exactly that.

RESULT: IT LOSES. Every configuration is worse -- room 93.42 -> 90.62 at
blend 0.7, and 86.50 -> 86.26 for the hand decoder alone. v_hat converged
correctly (median 3.34-3.69 against the measured 3.79), so the tracker worked
and the idea still failed.

The measurement was right and the inference from it was wrong. Step sizes vary
6x WITHIN the pooled distribution (p10 1.59, p90 9.49 px/frame). With
sigma_px=18 and a typical 5-frame gap, a prior centred at mu=6 sits ~13 px
short of the median step, which is well inside sigma, so the true position is
barely penalised. Centring on the correct median instead penalises the many
SHORT steps. fwd_px=6.0 is not an under-set constant, it is a conservative one
-- "expect little movement, let objectness decide" -- and given the variance,
conservative beats accurate. The heavy tail does the real work.

That also retracts the backward-step story: if mu being small were the cause of
stepping backward six times too often, correcting it would have helped.

Kept because the negative is informative, and because the same tempo estimate
may still work as a FEATURE, where the model can weight it conditionally
instead of the prior committing to it on every frame.

WHAT THIS DOES
--------------
Cont (2010) couples a tempo agent to the position agent so the expected
duration of the next event is tracked, not assumed. This is the small version:
the decoder keeps a per-piece EMA of the px/frame it has actually been
travelling, and the prior's mean becomes v_hat * dt.

NO LEAKAGE. v_hat is estimated online from the decoder's own steps within the
piece being decoded, warm-started from the shipped constant and only taking
over after `min_n` observations. Nothing measured on the test set is baked in
-- the 3.79 above motivated the experiment, it is not used as a value.

Updates are gated: a step only informs tempo if it is forward and plausible.
During a lock-loss episode the positions are wrong, and letting those steps
corrupt the tempo estimate would turn one failure into two.
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


def rollout(model, pages, blend=0.7, lam=1.0, topk=256, mu_pow=1.0,
            alpha=0.0, min_n=3, vmax=40.0):
    """alpha=0 keeps the shipped constant-tempo prior exactly."""
    hit = tot = 0
    vs = []
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
        feats = p.get('feat')
        v_hat, n_obs = FWD / REF, 0
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
                    dt = float(dfr or REF)
                    if alpha > 0 and n_obs >= min_n:
                        mu = v_hat * dt                  # tracked tempo
                    else:
                        mu = FWD * np.clip(dt / REF, 0.2, 8.0) ** mu_pow
                    hand = lo + lam * prior_logp(cs[:, 0] - x_prev, mu, SIG, JUMP)
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            xn = float(cs[j, 0])
            # GATED UPDATE: only a forward, plausible step informs tempo. A
            # lost tracker produces wild steps, and letting them in would turn
            # one failure into a corrupted tempo estimate for the rest of the
            # piece.
            if alpha > 0 and x_prev is not None and dfr:
                v_obs = (xn - x_prev) / float(dfr)
                if 0.0 < v_obs < vmax:
                    v_hat = (1 - alpha) * v_hat + alpha * v_obs
                    n_obs += 1
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = xn, float(cs[j, 1]), fr
        vs.append(v_hat)
    return 100.0 * hit / max(tot, 1), tot, float(np.median(vs))


_G = {}


def _init(ckpt, dumps):
    _G['model'] = load_ckpt(ckpt)[0]
    for nm, d in dumps.items():
        _G[nm] = load_with_feat(d)


def _run(job):
    which, blend, alpha, min_n = job
    acc, n, v = rollout(_G['model'], _G[which], blend=blend, alpha=alpha,
                        min_n=min_n)
    return which, blend, alpha, min_n, acc, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    dumps = {'do': '/scratch/pmohseni/omr/candf/do.npz',
             'room': '/scratch/pmohseni/omr/candf256/room.npz'}
    alphas, mins = [0.05, 0.1, 0.2, 0.4], [2, 5]
    jobs = [('do', b, 0.0, 3) for b in (0.0, 0.7)]
    jobs += [('do', b, al, mn) for b, al, mn
             in itertools.product((0.0, 0.7), alphas, mins)]
    with Pool(a.procs, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        res = pool.map(_run, jobs)

    # 86.50 and 91.44 are ROOM constants. Checking a `do` number against them
    # printed a MISMATCH that meant nothing; the do controls are 89.06 and
    # 95.01 and are only comparable to each other.
    for b in (0.0, 0.7):
        ctrl = [r for r in res if r[1] == b and r[2] == 0.0][0]
        print(f'\n=== blend {b} === control (constant tempo) on do: '
              f'{ctrl[4]:.2f}  (do control, NOT the room gate)')
        print(f'{"alpha":>6s} {"min_n":>6s} {"do":>7s} {"delta":>7s} {"v_hat":>7s}')
        for _, _, al, mn, acc, v in sorted(
                [r for r in res if r[1] == b and r[2] > 0], key=lambda r: -r[4]):
            print(f'{al:6.2f} {mn:6d} {acc:7.2f} {acc - ctrl[4]:+7.2f} {v:7.2f}')

    best = max([r for r in res if r[2] > 0 and r[1] == 0.7], key=lambda r: r[4])
    print(f'\n`do` picks alpha={best[2]} min_n={best[3]} at blend 0.7\n'
          f'=== the one look at room ===')
    with Pool(3, initializer=_init, initargs=(a.ckpt, dumps)) as pool:
        rr = pool.map(_run, [('room', 0.7, 0.0, 3), ('room', 0.7, best[2], best[3]),
                             ('room', 0.0, best[2], best[3])])
    for _, b, al, mn, acc, v in rr:
        what = 'constant tempo' if al == 0 else f'tracked (alpha={al}, min_n={mn})'
        print(f'  blend {b}  {what:32s} room {acc:6.2f}   median v_hat {v:.2f}')


if __name__ == '__main__':
    main()
