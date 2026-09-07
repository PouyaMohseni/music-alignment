"""vel_p8 vs ir_only on the HELD-OUT 80 pieces, where the sample size can resolve it.

On room the difference is +1.69 with a 95% CI of [-0.95, +3.62]: unresolvable,
and unresolvable for a structural reason rather than a lack of effect. 25 pages
collapse to 16 independent pieces, and each page's outcome is close to binary on
whether the tracker survives one lock-loss episode, so the variance swamps two
points no matter how good the scorer is.

This split has 230 pages over 80 pieces, five times the clusters, and shares no
piece with room OR with training. Clean, it sits at 97.3% -- a ceiling, useless
for discriminating anything -- so the comparison runs on the noise-degraded
tiers, where there is headroom.

WHAT THIS DOES AND DOES NOT TEST. It tests STATISTICAL generalization well: new
music, many more clusters. It tests ACOUSTIC generalization poorly, because
added noise is not a room mic, and the most reliable pattern in this project is
that mechanisms which help on synthetic hurt on room (five search mechanisms,
all of them). A win here is therefore evidence the effect is real, NOT evidence
it transfers to room -- room remains the only test for that.

Both arms read ~0.8 low against the harness because these dumps carry features
for the top 128 candidates only. It is the same handicap for both, so the
PAIRED quantity is unaffected, which is the quantity being reported.
"""
from __future__ import annotations

import argparse
import gc
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.offline_decode import TH, prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt


def rollout_hits(model, pages, blend=0.7, lam=1.0, fwd=6.0, sigma=18.0,
                 jump=-6.0, ref=5.0, mu_pow=1.0, topk=256):
    """Per-onset correctness plus the page each onset came from."""
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
                    k = np.clip((dfr or ref) / ref, 0.2, 8.0) ** mu_pow
                    hand = lo + lam * prior_logp(cs[:, 0] - x_prev, fwd * k,
                                                 sigma, jump)
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hits.append(abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH)
            page_of.append(p['name'])
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev, f_prev = float(cs[j, 0]), float(cs[j, 1]), fr
    return np.array(hits, bool), np.array(page_of)


def piece_of(pages):
    """Strip the trailing _page_N so pages of one piece share a cluster."""
    return np.array(['_page_'.join(p.split('_page_')[:-1]) or p for p in pages])


def paired_bootstrap(hb, hc, cluster, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    names = np.unique(cluster)
    idx = {n: np.flatnonzero(cluster == n) for n in names}
    d = []
    for _ in range(n_boot):
        take = np.concatenate([idx[n] for n in rng.choice(names, len(names), True)])
        d.append(100.0 * (hc[take].mean() - hb[take].mean()))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/scratch/pmohseni/omr/scorer/ir_only.pt')
    ap.add_argument('--cand', default='/scratch/pmohseni/omr/scorer/grid/vel_p8.pt')
    ap.add_argument('--snrs', default='12,6,3,0.5')
    a = ap.parse_args()

    mb = load_ckpt(a.base)[0]
    mc = load_ckpt(a.cand)[0]
    print(f'baseline {a.base.split("/")[-1]}   candidate {a.cand.split("/")[-1]}\n')
    print(f'{"tier":>8s} {"pieces":>7s} {"onsets":>7s} {"ir_only":>8s} '
          f'{"vel_p8":>7s} {"delta":>7s} {"95% CI":>18s} {"p":>7s}')

    for snr in a.snrs.split(','):
        path = f'/scratch/pmohseni/omr/candhv/valid_snr{snr}.npz'
        try:
            pages = load_with_feat(path)
        except Exception as e:
            print(f'{"snr" + snr:>8s}  load failed: {e}')
            continue
        hb, pg = rollout_hits(mb, pages)
        hc, _ = rollout_hits(mc, pages)
        cl = piece_of(pg)
        m, lo, hi, p = paired_bootstrap(hb, hc, cl)
        star = '  RESOLVED' if lo > 0 or hi < 0 else '  n.s.'
        print(f'{"snr" + snr:>8s} {len(np.unique(cl)):7d} {len(hb):7d} '
              f'{100.0 * hb.mean():8.2f} {100.0 * hc.mean():7.2f} {m:+7.2f} '
              f'[{lo:+6.2f},{hi:+6.2f}] {p:7.4f}{star}')
        del pages
        gc.collect()


if __name__ == '__main__':
    main()
