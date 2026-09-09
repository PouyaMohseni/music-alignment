"""Committee tracking: N independent trajectories, arbitrated by evidence.

WHY THIS AND NOT THE THINGS THAT FAILED
---------------------------------------
Per-step ranking is done: |dx| alone reaches 96.8% top-1 against a 96.0% causal
ceiling. What is left is lock-loss -- 71% of lost onsets sit in contiguous runs
-- and every attempt to fix it inside ONE trajectory has failed:

  gated recovery   precision never exceeds 0.33 at a 5% base rate, so firing
                   destroys two good locks for every one it saves
  soft recovery    same information, gentler action, still loses
  DAgger           the oracle action is often unreachable from a lost state
  beam search      degrades monotonically in width; accumulated path score
                   freezes the leader and fresh evidence cannot overturn it
  ensembling       averaged SCORES while sharing ONE trajectory

That last one is the clue. Every model in the ensemble conditioned on the same
x_prev, so they could not diverge and the average could not help. Yet the
per-page table shows different fits failing on DIFFERENT pages (+27.4 on one,
-4.3 on another). The diversity is real; the ensemble design discarded it.

So: give each model its own trajectory and arbitrate between them.

WHY THE WEAK DETECTOR IS ENOUGH HERE
------------------------------------
obj_chosen separates lost from fine at AUC 0.805 but only 0.33 precision, which
is fatal for a binary gate. Arbitration does not need a binary decision: it
needs to RANK which of N trackers is currently on music, and a 0.805 ranking
signal does that well. The same evidence that could not answer "am I lost"
can answer "which of us is least lost".

Two coupling modes:
  independent  each tracker keeps its own path; the committee only chooses what
               to EMIT. Maximum diversity, but a lost tracker stays lost.
  resample     after arbitration, trackers far from the winner and scoring
               badly are pulled to it -- a particle filter's resampling step,
               which lets a lost tracker rejoin instead of being dead weight.

Headroom is 2.6 points (93.4 against the 96.0 causal ceiling), so this cannot
be transformative. It is the last mechanism the evidence actually supports.
"""
from __future__ import annotations

import argparse
import glob
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


class Tracker:
    """One greedy decoder with its own history and its own prior constants."""

    def __init__(self, model, fwd=FWD, sig=SIG, blend=0.7, lam=1.0, decay=0.6):
        self.m, self.fwd, self.sig, self.blend, self.lam = model, fwd, sig, blend, lam
        self.decay = decay
        self.reset()

    def reset(self):
        self.x = self.y = self.x2 = self.f = self.f2 = None
        self.ev = 0.0          # smoothed objectness of what it has been picking

    def step(self, cs, bar, sys_, fr, ntot, ff):
        dfr = fr - self.f if self.f is not None and fr > self.f else None
        dfp = (self.f - self.f2 if self.f is not None and self.f2 is not None
               and self.f > self.f2 else None)
        f = build(cs, bar, sys_, self.x, self.y, dfr, ntot=ntot,
                  use_abs_obj=self.m.use_abs_obj, x_prev2=self.x2,
                  dframes_prev=dfp)[:, :self.m.nf]
        with torch.no_grad():
            s = self.m(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()
        if self.blend < 1.0:
            lo = np.log(np.clip(cs[:, 4], 1e-8, None))
            if self.x is None:
                hand = lo
            else:
                k = np.clip((dfr or REF) / REF, 0.2, 8.0)
                hand = lo + self.lam * prior_logp(cs[:, 0] - self.x,
                                                  self.fwd * k, self.sig, JUMP)
            s = self.blend * s + (1.0 - self.blend) * hand
        j = int(np.argmax(s))
        # smoothed evidence: low objectness sustained means this tracker is
        # somewhere with no music, which is what AUC 0.805 measures
        self.ev = self.decay * self.ev + (1 - self.decay) * float(cs[j, 4])
        return j, fr

    def commit(self, cs, j, fr):
        self.x2, self.f2 = self.x, self.f
        self.x, self.y, self.f = float(cs[j, 0]), float(cs[j, 1]), fr


def run(trackers, pages, mode='independent', pull=0.0, warm=5):
    hit = tot = 0
    for p in pages:
        for t in trackers:
            t.reset()
        feats = p.get('feat')
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:256]
            tot += 1
            fr = int(p['frame'][i])
            ffs = {}
            for t in trackers:
                if t.m.fenc is not None and feats is not None:
                    key = t.m.featdim
                    if key not in ffs:
                        fv = feats[i].astype(np.float32)
                        if fv.shape[0] < cs.shape[0]:
                            fv = np.vstack([fv, np.zeros(
                                (cs.shape[0] - fv.shape[0], key), np.float32)])
                        ffs[key] = torch.from_numpy(fv[:cs.shape[0]]).unsqueeze(0)
            picks = [t.step(cs, p['bar'][i], p['sys'][i], fr, int(p['ntot'][i]),
                            ffs.get(t.m.featdim)) for t in trackers]
            # ARBITRATE: emit the tracker currently best supported by evidence.
            # Ranking N trackers is the job a 0.805 AUC signal can do; deciding
            # "am I lost" at 0.33 precision is the job it cannot.
            w = np.array([t.ev for t in trackers])
            lead = int(np.argmax(w)) if i >= warm else 0
            j_lead = picks[lead][0]
            hit += abs(float(cs[j_lead, 5]) - float(p['t_gt'][i])) <= TH
            for t, (j, f_) in zip(trackers, picks):
                t.commit(cs, j, f_)
            if mode == 'resample' and pull > 0 and i >= warm:
                xl = trackers[lead].x
                for t in trackers:
                    if t is not trackers[lead] and t.ev < pull * trackers[lead].ev:
                        t.x, t.y = xl, trackers[lead].y   # rejoin the consensus
                        t.x2, t.f2 = None, None
    return 100.0 * hit / max(tot, 1), tot


M = '/scratch/pmohseni/omr/scorer'
_G = {}


def _init(dump):
    _G['pages'] = load_with_feat(dump)
    _G['models'] = {p: load_ckpt(p)[0] for p in
                    [f'{M}/grid/vel_p8.pt'] + sorted(glob.glob(f'{M}/seeds/vel_p8_s*.pt'))}


def _run(job):
    kind, n, mode, pull, decay = job
    paths = list(_G['models'])[:n]
    if kind == 'seeds':
        tk = [Tracker(_G['models'][p], decay=decay) for p in paths]
    else:   # one model, diversified by prior constants instead
        m = _G['models'][f'{M}/grid/vel_p8.pt']
        tk = [Tracker(m, fwd=f, sig=s, decay=decay) for f, s in
              list(itertools.product((3.0, 6.0, 12.0), (12.0, 18.0, 30.0)))[:n]]
    acc, tot = run(tk, _G['pages'], mode=mode, pull=pull)
    return kind, n, mode, pull, decay, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/scratch/pmohseni/omr/candf256/room.npz')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    jobs = [('seeds', 1, 'independent', 0.0, 0.6)]          # control = vel_p8
    for kind, n in itertools.product(('seeds', 'prior'), (3, 5)):
        for mode, pull in (('independent', 0.0), ('resample', 0.5),
                           ('resample', 0.8)):
            jobs.append((kind, n, mode, pull, 0.6))
    with Pool(a.procs, initializer=_init, initargs=(a.dump,)) as pool:
        res = pool.map(_run, jobs)
    ctrl = [r for r in res if r[1] == 1][0]
    print(f'control (single vel_p8, this code path): {ctrl[5]:.2f}')
    print('  gate: must equal 93.42 for the committee code to be trusted\n')
    print(f'{"diversity":9s} {"N":>2s} {"mode":>12s} {"pull":>5s} {"acc":>7s} {"delta":>7s}')
    for kind, n, mode, pull, dec, acc in sorted(res, key=lambda r: -r[5]):
        if n == 1:
            continue
        print(f'{kind:9s} {n:2d} {mode:>12s} {pull:5.1f} {acc:7.2f} '
              f'{acc - ctrl[5]:+7.2f}')


if __name__ == '__main__':
    main()
