"""The cross-validated decoder at all five published thresholds, not just 0.5 s.

lopo_new.py reports the primary figure alone, but Table 1 of both
[henkel2021frontiers] and [yang2026coda] is a row of five, and a row that
reports one cell and leaves four blank invites the reader to assume the others
are worse. Same protocol exactly: the four constants are fitted on 15 pieces
and the sixteenth is decoded with them, pooled over onsets.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import piece_of
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand
from extensions.analysis.offline_decode import prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt

FPS = 22050 / 1102
TH = (0.05, 0.1, 0.5, 1.0, 5.0)
BLENDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID = list(itertools.product((3.0, 4.5, 6.0, 8.0, 10.0), (12.0, 18.0, 27.0),
                              (-4.0, -6.0, -8.0)))


def errors(model, pages, blend, pr, topk=256):
    """Absolute error in seconds at every scored onset, in page order."""
    out = []
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = v_hat = None
        feats = p.get('feat')
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            fr = int(p['frame'][i])
            dfr = fr - f_prev if f_prev is not None and fr > f_prev else None
            dfp = (f_prev - f_prev2 if f_prev is not None
                   and f_prev2 is not None and f_prev > f_prev2 else None)
            f = build(cs, p['bar'][i], p['sys'][i], x_prev, y_prev, dfr,
                      ntot=int(p['ntot'][i]), use_abs_obj=model.use_abs_obj,
                      x_prev2=x_prev2, dframes_prev=dfp, v_hat=v_hat)[:, :model.nf]
            ff = None
            if model.fenc is not None and feats is not None:
                fv = feats[i].astype(np.float32)
                if fv.shape[0] < cs.shape[0]:
                    fv = np.vstack([fv, np.zeros((cs.shape[0] - fv.shape[0],
                                                  model.featdim), np.float32)])
                ff = torch.from_numpy(fv[:cs.shape[0]]).unsqueeze(0)
            with torch.no_grad():
                s = model(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()
            lo = np.log(np.clip(cs[:, 4], 1e-8, None))
            if x_prev is None:
                hand = lo
            else:
                k = np.clip((dfr or 5.0) / 5.0, 0.2, 8.0)
                hand = lo + prior_logp(cs[:, 0] - x_prev, pr[0] * k, pr[1], pr[2])
            j = int(np.argmax(blend * s + (1.0 - blend) * hand))
            out.append(abs(float(cs[j, 5]) - float(p['t_gt'][i])) / FPS)
            xn = float(cs[j, 0])
            if x_prev is not None and dfr:
                vo = (xn - x_prev) / float(dfr)
                if 0.0 < vo < 40.0:
                    v_hat = vo if v_hat is None else 0.8 * v_hat + 0.2 * vo
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = xn, float(cs[j, 1]), fr
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--ckpt', required=True, help='glob over seeds')
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    pages = load_with_feat(a.dump)
    arg, pg = rollout_argmax(pages)
    pieces = piece_of(pg)
    names = sorted(set(pieces))

    def macro(hits, held):
        return np.mean([hits[pieces == q].mean() for q in names if q != held])

    hand = {pr: rollout_hand(pages, fwd=pr[0], sigma=pr[1], jump=pr[2])[0]
            for pr in GRID}
    fold_prior = {p: max(GRID, key=lambda k: macro(hand[k], p)) for p in names}

    def row(label, err):
        print(f'{label:34s} ' + '  '.join(f'{np.mean(err <= t):.3f}' for t in TH),
              flush=True)

    def hit_row(label, hits):
        """rollout_argmax and rollout_hand return a hit/miss at 0.5 s only, so
        the other four columns of these two rows would be that same number
        wearing a different hat. Print it once and dash the rest rather than
        four copies a reader would take for a threshold sweep."""
        v = f'{hits.mean():.3f}'
        print(f'{label:34s} ' + '  '.join(v if t == 0.5 else '  ---' for t in TH),
              flush=True)

    print(f'{a.label or a.dump}   {len(arg)} onsets / {len(names)} pieces')
    print(f'{"":34s} ' + '  '.join(f'{t:>5}' for t in TH))
    hit_row('argmax', arg)
    lo = np.zeros(len(pieces), bool)
    for p in names:
        lo[pieces == p] = hand[fold_prior[p]][pieces == p]
    hit_row('+ prior (cross-validated)', lo)

    per_seed = []
    for ck in sorted(glob.glob(a.ckpt)):
        model = load_ckpt(ck)[0]
        cache = {}
        for pr in sorted(set(fold_prior.values())):
            for b in BLENDS:
                cache[(pr, b)] = errors(model, pages, b, pr)
        picked = np.zeros(len(pieces))
        for p in names:
            pr = fold_prior[p]
            b = max(BLENDS, key=lambda v: macro(cache[(pr, v)] <= 0.5, p))
            picked[pieces == p] = cache[(pr, b)][pieces == p]
        per_seed.append(picked)
        row(f'+ scorer  {ck.split("/")[-1]}', picked)
    if per_seed:
        m = np.stack([[np.mean(e <= t) for t in TH] for e in per_seed])
        print(f'{"+ scorer  MEAN of " + str(len(per_seed)) + " seeds":34s} '
              + '  '.join(f'{v:.3f}' for v in m.mean(0)))


if __name__ == '__main__':
    main()
