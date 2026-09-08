"""Per-PIECE numbers for all four arms, from one dump, for the artifact table."""
import sys, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import rollout_hits, piece_of
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand
from extensions.heads.cand_scorer import load as load_ckpt
M = '/scratch/pmohseni/omr/scorer'
pages = load_with_feat('/scratch/pmohseni/omr/candf256/room.npz')
arms = {}
arms['argmax'], pg = rollout_argmax(pages)
arms['hand'], _ = rollout_hand(pages)
arms['ir_only'], _ = rollout_hits(load_ckpt(f'{M}/ir_only.pt')[0], pages)
arms['vel_p8'], _ = rollout_hits(load_ckpt(f'{M}/grid/vel_p8.pt')[0], pages)
cl = piece_of(pg)
rows = []
for p in sorted(set(cl)):
    m = cl == p
    rows.append((p, int(m.sum()),
                 *[100.0 * arms[k][m].mean() for k in
                   ('argmax', 'hand', 'ir_only', 'vel_p8')]))
rows.sort(key=lambda r: -(r[5] - r[2]))
print(f'{"piece":52s} {"n":>5s} {"argmax":>7s} {"hand":>7s} {"ir_only":>8s} {"vel_p8":>7s} {"d":>7s}')
for r in rows:
    print(f'{r[0][:52]:52s} {r[1]:5d} {r[2]:7.1f} {r[3]:7.1f} {r[4]:8.1f} {r[5]:7.1f} {r[5]-r[2]:+7.1f}')
tot = [100.0 * arms[k].mean() for k in ('argmax','hand','ir_only','vel_p8')]
print(f'\n{"ALL":52s} {len(cl):5d} {tot[0]:7.2f} {tot[1]:7.2f} {tot[2]:8.2f} {tot[3]:7.2f} {tot[3]-tot[0]:+7.2f}')
imp = sum(1 for r in rows if r[5] > r[2]); print(f'pieces improved over argmax: {imp}/{len(rows)}')
