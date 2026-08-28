"""Decode dumped candidates offline, so ideas can be tested in milliseconds.

Every decoder variant so far cost one eight-minute eval.py run per setting, so a
sweep meant a job and a wait, and only a handful of ideas ever got tried. The
candidates and the ground truth are already dumped; running a decision rule over
them is pure arithmetic.

THE RULE THAT MAKES THIS SAFE: this file must reproduce the harness's own number
for the shipped decoder before any variant it reports means anything. `verify`
does exactly that and refuses to go on if it does not match.

It also unlocks decoders eval.py structurally cannot host. A fixed-lag smoother
has to see L frames past the one it is answering for, but get_max_box must
return that frame's box on the call for that frame -- so lookahead is not
expressible inside the harness at all, only over a dump.
"""
from __future__ import annotations

import glob
import numpy as np

FPS = 22050.0 / 1102.0
TH = 0.5 * FPS


def load(paths):
    """-> list of pages, each with per-frame arrays and a ragged candidate list."""
    pages = []
    for p in sorted(sum([glob.glob(x) for x in np.atleast_1d(paths)], [])):
        z = np.load(p, allow_pickle=False)
        for nm in sorted({k.split('||')[0] for k in z.files}):
            lens = z[f'{nm}||lens']
            if lens.size == 0:
                continue
            flat = z[f'{nm}||cand']
            off = np.concatenate([[0], np.cumsum(lens)])
            pages.append(dict(
                name=nm, frame=z[f'{nm}||frame'], t_gt=z[f'{nm}||t_gt'],
                ntot=z[f'{nm}||ntot'], bar=z[f'{nm}||bar'], sys=z[f'{nm}||sys'],
                cand=[flat[off[i]:off[i + 1]] for i in range(len(lens))]))
    return pages


def prior_logp(d, mu, sigma, jump):
    return np.maximum(-0.5 * ((d - mu) / sigma) ** 2, jump)


def greedy(page, lam=1.0, fwd=6.0, sigma=18.0, jump=-6.0, ref=5.0, mu_pow=1.0,
           topk=256, lookahead=0, win_topk=32):
    """The shipped rule at lookahead=0. lookahead=L runs a fixed-lag smoother:
    the position emitted for frame t is chosen by a DP over frames t..t+L, which
    lets a wrong commitment be overruled by evidence that arrives just after it.
    Costs L steps of latency -- 0.25 s at L=5, acceptable for page turning."""
    C = page['cand']
    n = len(C)
    out = np.full(n, np.nan)
    prev_x, prev_f = None, None
    t = 0
    while t < n:
        c = C[t]
        if c.shape[0] == 0:
            t += 1
            continue
        L = min(lookahead, n - 1 - t)
        # score() gives the log-score of every candidate at step s given a position
        def step(s, x_from, f_from):
            cs = C[s][:topk]
            if cs.shape[0] == 0:
                return None, None, None
            lo = np.log(np.clip(cs[:, 4], 1e-8, None))
            if x_from is None:
                return lo, cs[:, 0], cs
            df = max(int(page['frame'][s] - f_from), 1)
            sc = np.clip(df / ref, 0.2, 8.0) ** mu_pow
            return lo + lam * prior_logp(cs[:, 0] - x_from, fwd * sc, sigma, jump), cs[:, 0], cs
        if L <= 0:
            sc, xs, cs = step(t, prev_x, prev_f)
            j = int(np.argmax(sc))
            out[t] = cs[j, 5]
            prev_x, prev_f = float(xs[j]), int(page['frame'][t])
            t += 1
            continue
        # forward DP over the window, then commit only the first step
        sc, xs, cs = step(t, prev_x, prev_f)
        best, bx, bf = sc.copy(), xs, page['frame'][t]
        back = [np.arange(len(sc))]
        for s in range(t + 1, t + L + 1):
            # the window only has to answer "does the near future support this
            # choice", and the emission itself still ranks all `topk`. Widening
            # the window past ~32 buys nothing and makes the sweep O(n L K^2)
            # slow enough not to finish, which is how the first version died.
            nxt = C[s][:win_topk]
            if nxt.shape[0] == 0:
                back.append(None)
                continue
            df = max(int(page['frame'][s] - bf), 1)
            k = np.clip(df / ref, 0.2, 8.0) ** mu_pow
            lo = np.log(np.clip(nxt[:, 4], 1e-8, None))
            tr = prior_logp(nxt[None, :, 0] - bx[:, None], fwd * k, sigma, jump)
            tot = best[:, None] + lo[None, :] + lam * tr
            arg = np.argmax(tot, axis=0)
            best = tot[arg, np.arange(tot.shape[1])]
            back.append(arg)
            bx, bf = nxt[:, 0], page['frame'][s]
        # trace the best window path back to its first step
        j = int(np.argmax(best))
        for a in reversed(back[1:]):
            if a is not None:
                j = int(a[j])
        cs0 = C[t][:topk]
        out[t] = cs0[j, 5]
        prev_x, prev_f = float(cs0[j, 0]), int(page['frame'][t])
        t += 1
    return out


def viterbi_lag(page, lag=0, lam=1.0, fwd=6.0, sigma=18.0, jump=-6.0, ref=5.0,
                mu_pow=1.0, topk=64):
    """Fixed-lag smoother: one forward pass, then emit frame a by tracing back
    from the best state at a+lag.

    The first attempt rebuilt a DP window at every frame, O(n L K^2), and did not
    finish. Keeping the whole forward table costs n*K floats and makes each
    emission a `lag`-step walk backwards, so the total is one O(n K^2) pass plus
    O(n lag) pointer chasing.

    lag=0 is greedy-with-a-prior. lag >= n is the offline optimum under this
    prior, so one function spans the entire latency/accuracy curve.
    """
    C = [c[:topk] for c in page['cand']]
    idx = [i for i, c in enumerate(C) if c.shape[0]]
    n = len(idx)
    out = np.full(len(C), np.nan)
    if n == 0:
        return out
    dps = [np.log(np.clip(C[idx[0]][:, 4], 1e-8, None))]
    bps = [None]
    for a in range(1, n):
        i, j = idx[a - 1], idx[a]
        df = max(int(page['frame'][j] - page['frame'][i]), 1)
        k = np.clip(df / ref, 0.2, 8.0) ** mu_pow
        tr = prior_logp(C[j][None, :, 0] - C[i][:, 0, None], fwd * k, sigma, jump)
        tot = dps[-1][:, None] + lam * tr
        arg = np.argmax(tot, axis=0)
        dps.append(tot[arg, np.arange(tot.shape[1])]
                   + np.log(np.clip(C[j][:, 4], 1e-8, None)))
        bps.append(arg)
    for a in range(n):
        b = min(a + lag, n - 1)
        j = int(np.argmax(dps[b]))
        for t in range(b, a, -1):
            j = int(bps[t][j])
        out[idx[a]] = C[idx[a]][j, 5]
    return out


def score(pages, fn=None, **kw):
    fn = fn or greedy
    hit = tot = 0
    for p in pages:
        pred = fn(p, **kw)
        ok = ~np.isnan(pred)
        hit += int((np.abs(pred[ok] - p['t_gt'][ok]) <= TH).sum())
        tot += int(ok.sum())
    return 100.0 * hit / max(tot, 1), tot


def verify(pages, expected, tol=0.05, **kw):
    got, n = score(pages, **kw)
    ok = abs(got - expected) <= tol
    print(f'  offline decoder reproduces the harness: {got:.2f} vs {expected:.2f} '
          f'over {n} onsets  -> {"OK" if ok else "MISMATCH"}')
    if not ok:
        raise SystemExit('offline decoder does not match the harness; nothing '
                         'measured over it can be trusted')
    return got
