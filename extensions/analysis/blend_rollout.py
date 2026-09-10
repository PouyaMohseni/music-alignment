"""Blended scorer rollout over a dump -- the deployed decision rule, offline.

The trainer's rollout scores with the LEARNED term alone, but everything we
report on room is `blend * learned + (1 - blend) * hand`. Ranking checkpoints by
scorer-only rollout therefore ranks a configuration we never ship. This
replicates ScorerDecoder.decode exactly over dumped candidates, so a proxy set
can be scored the same way room is.

It must reproduce the harness before it is used for anything: ir_only at blend
0.7 on the room dump has to come back 91.4, and blend 0.0 has to come back 86.5.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.offline_decode import TH, load, prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt


def rollout(model, pages, blend=0.7, lam=1.0, fwd=6.0, sigma=18.0, jump=-6.0,
            ref=5.0, mu_pow=1.0, topk=256, back=None, tempo_alpha=0.2,
            vmax=40.0):
    hit = tot = 0
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
        # tracked tempo, from this rollout's own choices. Models fitted before
        # the tempo features exist have model.nf <= 33 and never see it.
        v_hat = None
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
                      x_prev2=x_prev2, dframes_prev=dfp, v_hat=v_hat,
                      pitch=(p['pitch'][i] if p.get('pitch') is not None else None))
            f = f[:, :model.nf]
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
                    k = np.clip((dfr or ref) / ref, 0.2, 8.0) ** mu_pow
                    d = cs[:, 0] - x_prev
                    pr = prior_logp(d, fwd * k, sigma, jump)
                    if back is not None:
                        pr = np.maximum(-0.5 * ((d - fwd * k) / sigma) ** 2,
                                        np.where(d < 0, back, jump))
                    hand = lo + lam * pr
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            xn = float(cs[j, 0])
            if x_prev is not None and dfr:
                vo = (xn - x_prev) / float(dfr)
                if 0.0 < vo < vmax:
                    v_hat = vo if v_hat is None else \
                        (1 - tempo_alpha) * v_hat + tempo_alpha * vo
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = xn, float(cs[j, 1]), fr
    return 100.0 * hit / max(tot, 1), tot


def load_with_feat(paths):
    pages = load(paths)
    for pat in np.atleast_1d(paths):
        for fp in sorted(glob.glob(pat)):
            z = np.load(fp, allow_pickle=False)
            has_feat = any(k.endswith('||feat') for k in z.files)
            has_z = any(k.endswith('||z') for k in z.files)
            if not (has_feat or has_z):
                continue
            for nm in sorted({k.split('||')[0] for k in z.files}):
                fk, flk = f'{nm}||feat', f'{nm}||flens'
                if flk not in z.files:
                    for pg in pages:
                        if pg['name'] == nm and f'{nm}||z' in z.files:
                            pg['z'] = z[f'{nm}||z']
                    continue
                fl = z[flk]
                off = np.concatenate([[0], np.cumsum(fl)])
                flat = z[fk]
                for pg in pages:
                    if pg['name'] != nm:
                        continue
                    pg['feat'] = [flat[off[i]:off[i + 1]] for i in range(len(fl))]
                    # z as well: the pitch heads need the audio vector, and
                    # without it annotate_pieces silently attaches nothing.
                    # This MUST sit under the name check -- at the outer level
                    # it assigned the current piece's z to every page, so each
                    # page carried another piece's audio and the length
                    # mismatch was the only reason it was noticed.
                    if f'{nm}||z' in z.files:
                        pg['z'] = z[f'{nm}||z']
                        assert len(pg['z']) == len(pg['cand']), (
                            f"{nm}: {len(pg['z'])} z rows for "
                            f"{len(pg['cand'])} frames")
    return pages
