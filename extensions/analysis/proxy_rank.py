"""Rank checkpoints on each candidate proxy set, correlate against room."""
import sys
import numpy as np
import torch

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
torch.set_num_threads(1)
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load

M = '/scratch/pmohseni/omr/scorer'
# geometry-only checkpoints with a known ROOM number at blend 0.7
ROOM = {'ir_only': 91.4, 'noz_only': 91.0, 'noz_union': 91.0, 'z_only': 90.6,
        'vel_only': 91.5, 'prec_tau05': 91.8, 'prec_tau1': 90.2,
        'prec_tau2': 91.3, 'prec_th_only': 91.4}
SETS = {'room (sanity)': '/scratch/pmohseni/omr/cand_test/room.npz',
        'do': '/scratch/pmohseni/omr/cand_test/do.npz',
        'rp_synth': '/scratch/pmohseni/omr/cand_test/rp_synth.npz',
        'synth valid': '/scratch/pmohseni/omr/cand_ir/valid_c0.npz'}


def rank(x):
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def pear(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else float('nan')


m0, _ = load(f'{M}/ir_only.pt')
pages = load_with_feat(SETS['room (sanity)'])
for b, want in ((0.0, 86.5), (0.7, 91.4)):
    got, n = rollout(m0, pages, blend=b)
    ok = abs(got - want) < 0.15
    print(f'GATE blend {b}: {got:.2f} vs harness {want}  {"OK" if ok else "MISMATCH"}',
          flush=True)
    if not ok:
        raise SystemExit('offline rollout does not reproduce the harness')

names = sorted(ROOM)
res = {}
for sname, path in SETS.items():
    pg = load_with_feat(path)
    sc = []
    for nm in names:
        m, _ = load(f'{M}/{nm}.pt')
        v, _ = rollout(m, pg, blend=0.7)
        sc.append(v)
    res[sname] = np.array(sc)
    print(f'\n{sname}:', flush=True)
    for nm, v in sorted(zip(names, sc), key=lambda t: -t[1]):
        print(f'   {nm:<14}{v:6.2f}   (room {ROOM[nm]})', flush=True)

R = np.array([ROOM[n] for n in names])
print('\n%-16s %10s %10s   picks         -> room' % ('proxy set', 'Spearman', 'Pearson'))
for sname, sc in res.items():
    best = names[int(np.argmax(sc))]
    print('%-16s %+10.3f %+10.3f   %-13s %5.1f'
          % (sname, pear(rank(sc), rank(R)), pear(sc, R), best, ROOM[best]))
print(f'\nbest achievable by picking on room itself: {max(ROOM.values())}')
