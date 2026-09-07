"""Average several selectors' scores before the argmax.

Every mechanism tried so far changed the DECISION RULE -- beam width, Viterbi,
pooling, re-anchoring, lookahead, recovery -- and all of them lost on room.
Ensembling changes neither the rule nor the capacity of any model: it averages
the scores of independently fitted selectors and takes the same greedy argmax.
That makes it the one remaining cheap mechanism whose failure mode is not the
one that killed the others (paying a cost on every frame to fix rare events).

It should help for a reason specific to what we measured today. The models sit
within a point of each other on room (feat_wide 92.7, vel_feat 92.7, vel_p8
93.4) but they are different fits, and the errors that dominate are lock-loss
episodes, which are nearly binary per page. If different fits enter episodes on
different pages -- and the per-page table showed exactly that spread -- then
averaging cancels the independent part.

If it does NOT help, that says the models are making the SAME mistakes, which
is itself worth knowing: it would mean the remaining error is a property of the
candidates rather than of any selector, and no amount of re-fitting will move
it.
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import paired_bootstrap, piece_of
from extensions.analysis.offline_decode import TH, prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt


def rollout_ens(models, pages, blend=0.7, lam=1.0, fwd=6.0, sigma=18.0,
                jump=-6.0, ref=5.0, mu_pow=1.0, topk=256):
    hits, page_of = [], []
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
        feats = p.get('feat')
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            fr = int(p['frame'][i])
            dfr = fr - f_prev if f_prev is not None and fr > f_prev else None
            dfp = (f_prev - f_prev2
                   if f_prev is not None and f_prev2 is not None and f_prev > f_prev2
                   else None)
            acc = None
            for m in models:
                f = build(cs, p['bar'][i], p['sys'][i], x_prev, y_prev, dfr,
                          ntot=int(p['ntot'][i]), use_abs_obj=m.use_abs_obj,
                          x_prev2=x_prev2, dframes_prev=dfp)[:, :m.nf]
                ff = None
                if m.fenc is not None and feats is not None:
                    fv = feats[i].astype(np.float32)
                    if fv.shape[0] < cs.shape[0]:
                        fv = np.vstack([fv, np.zeros(
                            (cs.shape[0] - fv.shape[0], m.featdim), np.float32)])
                    ff = torch.from_numpy(fv[:cs.shape[0]]).unsqueeze(0)
                with torch.no_grad():
                    s = m(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()
                # z-score per model so one with a wider score range cannot
                # dominate the average for reasons unrelated to being right
                s = (s - s.mean()) / max(s.std(), 1e-6)
                acc = s if acc is None else acc + s
            s = acc / len(models)
            if blend < 1.0:
                lo = np.log(np.clip(cs[:, 4], 1e-8, None))
                if x_prev is None:
                    hand = lo
                else:
                    k = np.clip((dfr or ref) / ref, 0.2, 8.0) ** mu_pow
                    hand = lo + lam * prior_logp(cs[:, 0] - x_prev, fwd * k,
                                                 sigma, jump)
                hand = (hand - hand.mean()) / max(hand.std(), 1e-6)
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hits.append(abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH)
            page_of.append(p['name'])
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = float(cs[j, 0]), float(cs[j, 1]), fr
    return np.array(hits, bool), np.array(page_of)


M = '/scratch/pmohseni/omr/scorer'
POOL = {'vel_p8': f'{M}/grid/vel_p8.pt', 'feat_wide': f'{M}/feat_wide.pt',
        'vel_feat': f'{M}/vel_feat.pt', 'vel_p64': f'{M}/grid/vel_p64.pt',
        'ir_only': f'{M}/ir_only.pt'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/scratch/pmohseni/omr/candf/room.npz')
    a = ap.parse_args()
    pages = load_with_feat(a.dump)
    cache = {k: load_ckpt(v)[0] for k, v in POOL.items()}

    # z-scoring changes the blend's units, so the single-model arm is NOT
    # bit-identical to the shipped rollout. Report it as its own control
    # rather than comparing against 92.60.
    single, pg = rollout_ens([cache['vel_p8']], pages)
    cl = piece_of(pg)
    base = 100.0 * single.mean()
    print(f'vel_p8 alone under this normalisation: {base:.2f}  '
          f'({len(single)} onsets)\n')
    print(f'{"combination":34s} {"pct":>6s} {"delta":>7s} {"95% CI":>18s}')
    combos = []
    for r in (2, 3):
        for c in itertools.combinations(
                ['vel_p8', 'feat_wide', 'vel_feat', 'vel_p64', 'ir_only'], r):
            if 'vel_p8' not in c:
                continue
            combos.append(c)
    for c in combos:
        h, _ = rollout_ens([cache[k] for k in c], pages)
        m, lo, hi, _ = paired_bootstrap(single, h, cl)
        tag = '  RESOLVED' if lo > 0 or hi < 0 else ''
        print(f'{"+".join(c):34s} {100.0 * h.mean():6.2f} {m:+7.2f} '
              f'[{lo:+6.2f},{hi:+6.2f}]{tag}', flush=True)


if __name__ == '__main__':
    main()
