"""Is the note index usable ONLINE, or does 87.2 depend on seeing the future?

The bins that gave 87.2 were pooled over the whole recording, so that figure is
an upper bound. Two causal alternatives:

  prescan  -- bins from the FIRST frame's candidates only. In deployment the
              score page is available before a note is played, so one forward
              pass over the page is legitimate; this approximates it.
  running  -- bins rebuilt from every candidate seen so far, updated as the
              piece proceeds. Strictly causal, no page pre-scan assumed.

If `running` holds most of the gain, the coordinate is genuinely available
online. If only the pooled version works, 87.2 is an artefact of hindsight and
should not be reported.
"""
import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.note_index import to_index
from extensions.analysis.offline_decode import TH, load, prior_logp

SETS = {'room': '/scratch/pmohseni/omr/cand_test/room.npz',
        'do': '/scratch/pmohseni/omr/cand_test/do.npz'}


def _cluster(x, bin_px):
    if x.size == 0:
        return np.zeros(0)
    x = np.sort(x)
    cut = np.nonzero(np.diff(x) > bin_px)[0]
    return np.array([g.mean() for g in np.split(x, cut + 1)])


def score(pages, mode='pooled', bin_px=8.0, fwd=0.5, sigma=1.5, jump=-6.0,
          lam=1.0, ref=5.0, mu_pow=1.0, topk=100000, min_obj=0.10):
    hit = tot = 0
    for p in pages:
        allx = [c[c[:, 4] >= min_obj, 0] for c in p['cand'] if c.shape[0]]
        pooled = _cluster(np.concatenate(allx), bin_px) if allx else np.zeros(0)
        first = _cluster(allx[0], bin_px) if allx else np.zeros(0)
        seen, ctr = [], (pooled if mode == 'pooled' else first)
        ip = fp = None
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            tot += 1
            if mode == 'running':
                seen.append(cs[cs[:, 4] >= min_obj, 0])
                ctr = _cluster(np.concatenate(seen), bin_px)
            pos = to_index(cs[:, 0], ctr)
            lo = np.log(np.clip(cs[:, 4], 1e-8, None))
            fr = int(p['frame'][i])
            if ip is None:
                s = lo
            else:
                k = np.clip(((fr - fp) if fp and fr > fp else ref) / ref,
                            0.2, 8.0) ** mu_pow
                s = lo + lam * prior_logp(pos - ip, fwd * k, sigma, jump)
            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            ip, fp = float(pos[j]), fr
    return 100.0 * hit / max(tot, 1), tot


print('%-8s %-10s %8s' % ('tier', 'mode', 'pct@0.5s'), flush=True)
for nm, path in SETS.items():
    pg = load(path)
    for mode in ('pooled', 'prescan', 'running'):
        v, n = score(pg, mode=mode)
        print('%-8s %-10s %8.1f' % (nm, mode, v), flush=True)
print('\n  pixel reference: room 86.5, do 89.1', flush=True)
print('  pooled is NOT causal and is an upper bound; running is strictly causal',
      flush=True)
