"""Rank the FEATURE-based selectors on `do`, the only proxy with signal.

feat_wide reached 94.0 on room but validation would have picked a 90.0 model
(Spearman -0.39), so 94.0 has never been defensible. `do` is real audio 3.6
points from room in difficulty and ranks checkpoints at Spearman +0.60, against
+0.30 for synthetic validation -- but cand_test carried no backbone features, so
it could not rank the feature models at all. It can now.

CAVEAT that has to travel with any number selected this way: `do` shares all 16
pieces with room. Piece identity leaks; only the acoustic condition differs. It
is leave-one-condition-out, not a clean holdout.

Gate: ir_only must return 91.4 on room at blend 0.7 and 86.5 at blend 0.0.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
torch.set_num_threads(1)
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load

M = '/scratch/pmohseni/omr/scorer'
ROOM_KNOWN = {'ir_only': 91.4, 'feat_base': 88.9, 'feat_small': 91.3,
              'feat_wide': 94.0, 'vel_feat': 90.0, 'vel_only': 91.5,
              'noz_only': 91.0, 'prec_tau05': 91.8}

room = load_with_feat('/scratch/pmohseni/omr/candf/room.npz')
do = load_with_feat('/scratch/pmohseni/omr/candf/do.npz')

m0, _ = load(f'{M}/ir_only.pt')
print('GATE on room (features dump):', flush=True)
for b, want in ((0.0, 86.5), (0.7, 91.4)):
    got, n = rollout(m0, room, blend=b)
    ok = abs(got - want) < 0.2
    print(f'  blend {b}: {got:.2f} vs {want}  {"OK" if ok else "MISMATCH"}  ({n} onsets)',
          flush=True)
    if not ok:
        raise SystemExit('gate failed; the features dump does not reproduce')

names = sorted(set(ROOM_KNOWN) | {p.split('/')[-1][:-3]
                                  for p in glob.glob(f'{M}/grid/*.pt')})
rows = []
print('\n%-14s %8s %8s %10s' % ('variant', 'do', 'room', 'known room'), flush=True)
for nm in names:
    p = f'{M}/{nm}.pt' if glob.glob(f'{M}/{nm}.pt') else f'{M}/grid/{nm}.pt'
    if not glob.glob(p):
        continue
    try:
        m, _ = load(p)
    except Exception as e:
        print(f'  {nm}: load failed ({e})', flush=True)
        continue
    d, _ = rollout(m, do, blend=0.7)
    r, _ = rollout(m, room, blend=0.7)
    rows.append((d, r, nm))
    print('%-14s %8.1f %8.1f %10s'
          % (nm, d, r, ROOM_KNOWN.get(nm, '-')), flush=True)

rows.sort(reverse=True)
print(f'\n  `do` picks : {rows[0][2]}  ->  room {rows[0][1]:.1f}', flush=True)
best_room = max(rows, key=lambda t: t[1])
print(f'  best on room: {best_room[2]}  ->  room {best_room[1]:.1f} '
      f'(picking on room itself, not defensible)', flush=True)
d = np.array([r[0] for r in rows]); r = np.array([r[1] for r in rows])
def rk(x):
    o = np.argsort(x); q = np.empty(len(x)); q[o] = np.arange(len(x)); return q
def pe(x, y):
    x = x - x.mean(); y = y - y.mean()
    s = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / s) if s else float('nan')
print(f'  Spearman(do, room) over {len(rows)} feature-capable models = '
      f'{pe(rk(d), rk(r)):+.3f}', flush=True)
