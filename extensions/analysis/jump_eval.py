"""Jump recovery over a spliced take: CODA's break mode, and two of our own.

WHAT IS BEING COMPARED
----------------------
Our decoder's transition prior is a floored Gaussian, so a jump costs a fixed
`jump` nats and the tracker resists it. That is structurally the row CODA labels
"w/o break" (.29 recovery at 1 s), and their break mode takes it to .78. Three
recovery rules are run over the identical dump:

  break   CODA's, ported. Waveform energy under a threshold for `hold` frames
          enters a break: the committed position freezes and the transition
          prior is suppressed. When energy returns the prior stays suppressed
          for a `grace` window, during which selection is by appearance alone,
          and the position reached at the end of it is committed.

  conf    ours, and it needs no silence. The prior is what pins the tracker to
          a stale position, and the giveaway is that the pinned candidate is far
          below the frame's best by objectness (AUC 0.805 for lost episodes in
          our own failure analysis). When the chosen candidate has been `delta`
          nats below the frame's best for `win` consecutive frames, the history
          is dropped for one frame and selection restarts from appearance.

  beam    ours, two hypotheses. One follows the prior, one ignores it entirely
          and takes the frame's best candidate. Both accumulate log objectness
          over a sliding `win` frames, and the prior-free track replaces the
          prior-following one when it has been ahead by `delta` over the whole
          window. Strictly more state than `conf` and it commits on evidence
          gathered rather than on a single frame.

`conf` and `beam` fire on the tracker's own disagreement, so they apply equally
to a repeat, a page turn, a performer error and an ordinary propagation failure,
which is 66-76% of our remaining error. Silence is CODA's cue and a good one for
repeats, but Nakamura's own count is 59 of 63, and a page turn has no break at
all. Running both against gap 8 and gap 0 dumps separates the two claims.

METRICS
-------
CODA's, with one substitution. They report SYSTEM recovery; we predict a
position on an unrolled strip and never emit a system index, so recovery is
"predicted position within `th` seconds of the truth", which is the same
quantity their post-jump column already uses. Reported per jump:

  rec@1, rec@2   fraction of jumps first tracked correctly within 1 s / 2 s
  lat            mean seconds to the first correct frame (censored at 5 s)
  post           fraction of ONSETS in the 5 s after the jump tracked within 1 s

The `post` column is the one directly comparable to their Table 2, where
CYOLO-SB scores .27 on the random subset and CODA .71.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.offline_decode import prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt

FPS = 22050 / 1102
REF = 5.0


def _score(model, cs, feats_i, bar, sysb, x_prev, y_prev, dfr, ntot,
           x_prev2, dfp, v_hat):
    f = build(cs, bar, sysb, x_prev, y_prev, dfr, ntot=ntot,
              use_abs_obj=model.use_abs_obj, x_prev2=x_prev2,
              dframes_prev=dfp, v_hat=v_hat)[:, :model.nf]
    ff = None
    if model.fenc is not None and feats_i is not None:
        fv = feats_i.astype(np.float32)
        if fv.shape[0] < cs.shape[0]:
            fv = np.vstack([fv, np.zeros((cs.shape[0] - fv.shape[0],
                                          model.featdim), np.float32)])
        ff = torch.from_numpy(fv[:cs.shape[0]]).unsqueeze(0)
    with torch.no_grad():
        return model(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()


def rollout(model, page, rms, mode='none', blend=0.7, fwd=10.0, sigma=18.0,
            jump=-8.0, topk=256, hold=3, grace=16, sil_rel=0.02, win=8,
            delta=1.5):
    """Per-frame predicted onset-frame position, and the frames scored."""
    n = len(page['cand'])
    pred = np.full(n, np.nan)
    feats = page.get('feat')
    x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
    v_hat = None
    state, since, run_hi = 'run', 0, 1e-9
    lo_hist, alt_hist = [], []
    alt_x = None                      # the prior-free hypothesis, for `beam`
    for i in range(n):
        c = page['cand'][i]
        if c.shape[0] == 0:
            continue
        cs = c[:topk]
        fr = int(page['frame'][i])
        dfr = fr - f_prev if f_prev is not None and fr > f_prev else None
        dfp = (f_prev - f_prev2 if f_prev is not None and f_prev2 is not None
               and f_prev > f_prev2 else None)
        lo = np.log(np.clip(cs[:, 4], 1e-8, None))
        best_lo = float(lo.max())

        if mode == 'break' and rms is not None:
            e = float(rms[i]) if i < len(rms) else 0.0
            run_hi = max(run_hi * 0.999, e)
            quiet = e < sil_rel * run_hi
            if state == 'run':
                since = since + 1 if quiet else 0
                if since >= hold:
                    state, since = 'break', 0
            elif state == 'break':
                if not quiet:
                    state, since = 'grace', 0
            else:
                since += 1
                if since >= grace:
                    state = 'run'
            if state == 'break':
                # position frozen, nothing committed, nothing scored against
                pred[i] = x_prev if x_prev is not None else np.nan
                continue

        free = (mode == 'break' and state == 'grace') or x_prev is None
        s = _score(model, cs, feats[i] if feats is not None else None,
                   page['bar'][i], page['sys'][i], None if free else x_prev,
                   None if free else y_prev, dfr, int(page['ntot'][i]),
                   None if free else x_prev2, dfp, None if free else v_hat)
        if free:
            tot = blend * s + (1.0 - blend) * lo
        else:
            k = np.clip((dfr or REF) / REF, 0.2, 8.0)
            hand = lo + prior_logp(cs[:, 0] - x_prev, fwd * k, sigma, jump)
            tot = blend * s + (1.0 - blend) * hand
        j = int(np.argmax(tot))

        if mode == 'conf' and not free:
            lo_hist.append(best_lo - float(lo[j]))
            if len(lo_hist) > win:
                lo_hist.pop(0)
            if len(lo_hist) == win and min(lo_hist) > delta:
                # pinned somewhere with no music: restart from appearance
                s2 = _score(model, cs, feats[i] if feats is not None else None,
                            page['bar'][i], page['sys'][i], None, None, dfr,
                            int(page['ntot'][i]), None, dfp, None)
                j = int(np.argmax(blend * s2 + (1.0 - blend) * lo))
                lo_hist.clear()
                v_hat = None
        elif mode == 'beam':
            a = int(np.argmax(lo)) if alt_x is None else None
            if alt_x is not None:
                sa = _score(model, cs, feats[i] if feats is not None else None,
                            page['bar'][i], page['sys'][i], alt_x, y_prev, dfr,
                            int(page['ntot'][i]), None, dfp, None)
                ka = np.clip((dfr or REF) / REF, 0.2, 8.0)
                ha = lo + prior_logp(cs[:, 0] - alt_x, fwd * ka, sigma, jump)
                a = int(np.argmax(blend * sa + (1.0 - blend) * ha))
            alt_hist.append(float(lo[a]) - float(lo[j]))
            if len(alt_hist) > win:
                alt_hist.pop(0)
            if len(alt_hist) == win and float(np.mean(alt_hist)) > delta:
                j, alt_x, v_hat = a, None, None
                alt_hist.clear()
            else:
                alt_x = float(cs[int(np.argmax(lo)), 0]) if alt_x is None \
                    else float(cs[a, 0])

        pred[i] = float(cs[j, 5])
        xn = float(cs[j, 0])
        if x_prev is not None and dfr:
            vo = (xn - x_prev) / float(dfr)
            if 0.0 < vo < 40.0:
                v_hat = vo if v_hat is None else 0.8 * v_hat + 0.2 * vo
        x_prev2, f_prev2 = x_prev, f_prev
        x_prev, y_prev, f_prev = xn, float(cs[j, 1]), fr
    return pred


def jump_metrics(pred, gt, onset, jumps, th=0.5, horizon=5.0):
    """CODA's three, per jump: recovery at 1 s and 2 s, latency, post-jump."""
    ok = np.abs(pred - gt) / FPS <= th
    rec1, rec2, lat, post = [], [], [], []
    H = int(round(horizon * FPS))
    for at, _src, _dst, _gap in jumps:
        at = int(at)
        seg = slice(at, min(at + H, len(ok)))
        w = np.flatnonzero(ok[seg])
        first = w[0] / FPS if w.size else np.nan
        rec1.append(bool(w.size and first <= 1.0))
        rec2.append(bool(w.size and first <= 2.0))
        lat.append(horizon if np.isnan(first) else first)
        m = onset[seg]
        post.append(float(np.mean(np.abs(pred[seg][m] - gt[seg][m]) / FPS <= 1.0))
                    if m.any() else np.nan)
    return (np.array(rec1, float), np.array(rec2, float),
            np.array(lat, float), np.array(post, float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--side', required=True)
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--modes', nargs='+',
                    default=['none', 'break', 'conf', 'beam'])
    ap.add_argument('--blend', type=float, default=0.7)
    ap.add_argument('--fwd', type=float, default=10.0)
    ap.add_argument('--sigma', type=float, default=18.0)
    ap.add_argument('--jump', type=float, default=-8.0)
    ap.add_argument('--win', type=int, default=8)
    ap.add_argument('--delta', type=float, default=1.5)
    ap.add_argument('--grace', type=int, default=16)
    ap.add_argument('--th', type=float, default=0.5)
    ap.add_argument('--argmax', action='store_true',
                    help='also the detector-only row, the CYOLO-SB baseline')
    a = ap.parse_args()

    pages = load_with_feat(a.dump)
    side = np.load(a.side, allow_pickle=False)
    have = {k.split('||')[0] for k in side.files}
    pages = [p for p in pages if p['name'] in have]
    njump = sum(len(side[p['name'] + '||jumps']) for p in pages)
    print(f'{len(pages)} pages, {njump} jumps, dump {os.path.basename(a.dump)}\n',
          flush=True)

    def report(label, per_page):
        r1 = np.concatenate([x[0] for x in per_page])
        r2 = np.concatenate([x[1] for x in per_page])
        lt = np.concatenate([x[2] for x in per_page])
        po = np.concatenate([x[3] for x in per_page])
        print(f'{label:28s} {r1.mean():5.2f} {r2.mean():5.2f} {lt.mean():6.2f} '
              f'  {np.nanmean(po):5.2f}', flush=True)

    print(f'{"method":28s} {"rec@1":>5s} {"rec@2":>5s} {"lat":>6s}   {"post":>5s}')

    if a.argmax:
        per = []
        for p in pages:
            n = len(p['cand'])
            pred = np.full(n, np.nan)
            for i, c in enumerate(p['cand']):
                if c.shape[0]:
                    pred[i] = float(c[int(np.argmax(c[:, 4])), 5])
            per.append(jump_metrics(pred, np.asarray(p['t_gt'], float),
                                    side[p['name'] + '||is_onset'],
                                    side[p['name'] + '||jumps'], a.th))
        report('argmax (detector only)', per)

    for ck in a.ckpt:
        model = load_ckpt(ck)[0]
        for mode in a.modes:
            per = []
            for p in pages:
                pred = rollout(model, p, side[p['name'] + '||rms'], mode=mode,
                               blend=a.blend, fwd=a.fwd, sigma=a.sigma,
                               jump=a.jump, win=a.win, delta=a.delta,
                               grace=a.grace)
                per.append(jump_metrics(pred, np.asarray(p['t_gt'], float),
                                        side[p['name'] + '||is_onset'],
                                        side[p['name'] + '||jumps'], a.th))
            report(f'{os.path.basename(ck)[:-3]} + {mode}', per)


if __name__ == '__main__':
    main()
