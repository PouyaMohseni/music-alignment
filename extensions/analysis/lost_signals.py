"""Is being LOST observable at decode time, without ground truth?

Everything downstream depends on this. The episode analysis showed 71% of lost
onsets sit in contiguous runs of >=5, so the failure has a shape worth
recovering from -- but a recovery can only be GATED if something visible to the
decoder distinguishes "lost" from "fine". Wider beams and re-anchoring both lost
on room precisely because they fired everywhere; a gate is the whole difference.

The hard part is that a lost tracker is usually LOCALLY CONSISTENT. Once x_prev
is wrong, the next step is still a small plausible displacement from it, so the
transition residual looks healthy. If that is all there is, no causal signal
separates the states and the gate cannot exist -- which is a real answer, and
better found now than after building the recovery.

Every signal here is causal: it uses the current frame's candidates and the
decoder's own past, never the future and never t_gt.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

# One thread. Unpinned, torch takes every core it can see, which on a login
# node is 64 of them at 520% CPU -- an AUP violation, and this rollout is
# serial over onsets anyway so the threads buy nothing.
torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.offline_decode import TH, prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt

SIGNALS = ['margin', 'smax', 'entropy', 'obj_chosen', 'obj_max', 'obj_rank',
           'dx_norm', 'resid', 'agree', 'hand_gap', 'ncand',
           'margin_m5', 'agree_m5', 'resid_m5', 'since_agree']


def _softmax(s):
    e = np.exp(s - s.max())
    return e / max(e.sum(), 1e-12)


def rollout_signals(model, pages, blend=0.7, lam=1.0, fwd=6.0, sigma=18.0,
                    jump=-6.0, ref=5.0, mu_pow=1.0, topk=256):
    rows, page_of = [], []
    for p in pages:
        x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
        feats = p.get('feat')
        hist_margin, hist_agree, hist_resid, since_agree = [], [], [], 0
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
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
                s_learn = model(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()

            lo = np.log(np.clip(cs[:, 4], 1e-8, None))
            if x_prev is None:
                hand = lo
                k = 1.0
            else:
                k = np.clip((dfr or ref) / ref, 0.2, 8.0) ** mu_pow
                hand = lo + lam * prior_logp(cs[:, 0] - x_prev, fwd * k, sigma, jump)
            s = blend * s_learn + (1.0 - blend) * hand if blend < 1.0 else s_learn
            j = int(np.argmax(s))

            srt = np.sort(s)[::-1]
            pr = _softmax(s)
            dx = 0.0 if x_prev is None else float(cs[j, 0]) - x_prev
            mu = fwd * k
            rows.append(dict(
                margin=float(srt[0] - srt[1]) if srt.size > 1 else 10.0,
                smax=float(srt[0]),
                entropy=float(-(pr * np.log(pr + 1e-12)).sum()),
                obj_chosen=float(cs[j, 4]),
                obj_max=float(cs[:, 4].max()),
                obj_rank=float((cs[:, 4] > cs[j, 4]).sum()),
                dx_norm=float(dx / max(mu, 1e-6)),
                resid=float(abs(dx - mu) / sigma) if x_prev is not None else 0.0,
                agree=float(j == int(np.argmax(hand))),
                hand_gap=float(hand.max() - hand[j]),
                ncand=float(cs.shape[0]),
                margin_m5=float(np.mean(hist_margin[-5:])) if hist_margin else 10.0,
                agree_m5=float(np.mean(hist_agree[-5:])) if hist_agree else 1.0,
                resid_m5=float(np.mean(hist_resid[-5:])) if hist_resid else 0.0,
                since_agree=float(since_agree),
                wrong=float(abs(float(cs[j, 5]) - float(p['t_gt'][i])) > TH)))
            page_of.append(p['name'])
            hist_margin.append(rows[-1]['margin'])
            hist_agree.append(rows[-1]['agree'])
            hist_resid.append(rows[-1]['resid'])
            since_agree = 0 if rows[-1]['agree'] else since_agree + 1
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = float(cs[j, 0]), float(cs[j, 1]), fr
    return rows, np.array(page_of)


def episodes(wrong, page, minlen=5):
    """Mark onsets inside a contiguous wrong-run of >= minlen, per page."""
    out = np.zeros(len(wrong), bool)
    for nm in np.unique(page):
        m = page == nm
        w = wrong[m].astype(bool)
        idx = np.flatnonzero(m)
        i = 0
        while i < len(w):
            if w[i]:
                j = i
                while j < len(w) and w[j]:
                    j += 1
                if j - i >= minlen:
                    out[idx[i:j]] = True
                i = j
            else:
                i += 1
    return out


def auc(x, y):
    y = y.astype(bool)
    if y.all() or not y.any():
        return float('nan')
    r = np.argsort(np.argsort(x)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--dump', default='/scratch/pmohseni/omr/candf/room.npz')
    ap.add_argument('--blend', type=float, default=0.7)
    ap.add_argument('--minlen', type=int, default=5)
    a = ap.parse_args()

    model, meta = load_ckpt(a.ckpt)
    pages = load_with_feat(a.dump)
    rows, page = rollout_signals(model, pages, blend=a.blend)
    wrong = np.array([r['wrong'] for r in rows])
    print(f'{len(rows)} onsets, pct@0.5s = {100.0 * (1 - wrong.mean()):.2f} '
          f'(rollout replica; the harness reads 0.8 higher because this dump '
          f'has features for only the top 128 candidates)')

    ep = episodes(wrong, page, a.minlen)
    print(f'wrong {int(wrong.sum())}  in episodes >={a.minlen}: {int(ep.sum())} '
          f'({100.0 * ep.sum() / max(wrong.sum(), 1):.0f}% of errors)\n')

    print(f'{"signal":12s} {"AUC(in episode)":>16s} {"AUC(wrong)":>11s}')
    order = []
    for k in SIGNALS:
        v = np.array([r[k] for r in rows])
        A, B = auc(v, ep), auc(v, wrong.astype(bool))
        order.append((abs(A - 0.5), k, A, B))
        print(f'{k:12s} {A:16.3f} {B:11.3f}')
    order.sort(reverse=True)
    print(f'\nmost separating: ' +
          ', '.join(f'{k} ({A:.3f})' for _, k, A, _ in order[:4]))
    best = order[0][0] + 0.5
    print('\nVERDICT: ' + (
        'a causal gate is possible -- at least one signal separates the states'
        if best >= 0.65 else
        'NO usable gate from these signals; a lost tracker looks locally normal'))


if __name__ == '__main__':
    main()
