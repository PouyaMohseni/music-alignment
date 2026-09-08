"""Pick ONE model and ONE number, without ever selecting on room.

Everything measured so far says the same thing: room has 16 independent pieces
and cannot resolve differences of a point or two, so choosing a model by its
room score is choosing noise. It also cannot measure seed variance -- its
apparent seed range of 2.05 shrank to sd 0.70 once measured on 80 pieces.

So the selection uses room for nothing at all:

  CONFIG      by held-out config MEAN over seeds. 80 pieces, no overlap with
              room or training, and a mean over draws rather than a best draw.
  CHECKPOINT  by the trainer's own VALIDATION metric, within the winning
              config only. Validation was always the right instrument for
              picking a draw; it just was not being used for it.
  ROOM        read once, at the end, for the winner.

Both configs are scored on FEATK=256 dumps because that is what the harness
supplies at inference. Scoring the 128-trained config on 128-capped dumps
would flatter it relative to how it actually runs.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.analysis.heldout_compare import (paired_bootstrap, piece_of,
                                                 rollout_hits)
from extensions.heads.cand_scorer import load as load_ckpt

M = '/scratch/pmohseni/omr/scorer'
CONFIGS = {
    'vel_p8 (trained@128)':
        [f'{M}/grid/vel_p8.pt'] + sorted(glob.glob(f'{M}/seeds/vel_p8_s*.pt')),
    'vel_p8_f256 (trained@256)':
        [f'{M}/f256/vel_p8_f256.pt'] + sorted(glob.glob(f'{M}/f256/vel_p8_f256_s*.pt')),
}


def valid_metric(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    ex = d.get('extra') or {}
    for k in ('valid_rollout', 'valid', 'valid_acc'):
        if k in ex:
            return float(ex[k])
    return float('nan')


def main():
    hv = '/scratch/pmohseni/omr/candhv256/valid_snr12.npz'
    if not glob.glob(hv):
        print(f'held-out FEATK=256 dump missing ({hv}); hv256 must finish first')
        return
    pages = load_with_feat(hv)
    hb, pg = rollout_hits(load_ckpt(f'{M}/ir_only.pt')[0], pages)
    cl = piece_of(pg)
    print(f'held-out snr12 @FEATK=256: {len(hb)} onsets, {len(np.unique(cl))} '
          f'pieces\nir_only baseline {100.0 * hb.mean():.2f}\n')

    summary = {}
    for name, paths in CONFIGS.items():
        paths = [p for p in paths if glob.glob(p)]
        if not paths:
            print(f'{name}: no checkpoints'); continue
        accs, deltas = [], []
        print(f'--- {name} ({len(paths)} seeds)')
        for p in paths:
            h, _ = rollout_hits(load_ckpt(p)[0], pages)
            m, lo, hi, pv = paired_bootstrap(hb, h, cl)
            accs.append(100.0 * h.mean()); deltas.append(m)
            print(f'   {p.split("/")[-1]:24s} {accs[-1]:6.2f} {m:+6.2f} '
                  f'[{lo:+5.2f},{hi:+5.2f}] valid={valid_metric(p):.4f}',
                  flush=True)
        a = np.array(accs)
        summary[name] = dict(paths=paths, heldout=accs, mean=a.mean(),
                             sd=a.std(ddof=1) if len(a) > 1 else 0.0,
                             delta=float(np.mean(deltas)), n=len(a))
        print(f'   CONFIG MEAN {a.mean():.2f}  sd {summary[name]["sd"]:.2f}  '
              f'delta vs ir_only {np.mean(deltas):+.2f}\n')

    if not summary:
        return
    win = max(summary, key=lambda k: summary[k]['mean'])
    print(f'=== CONFIG CHOSEN ON HELD-OUT: {win} '
          f'(mean {summary[win]["mean"]:.2f}, n={summary[win]["n"]}) ===')
    if len(summary) > 1:
        other = [k for k in summary if k != win][0]
        gap = summary[win]['mean'] - summary[other]['mean']
        pooled = max(np.hypot(summary[win]['sd'], summary[other]['sd']), 1e-9)
        print(f'    beats {other} by {gap:+.2f} (pooled seed sd {pooled:.2f})'
              + ('  -- inside seed noise, treat as a tie'
                 if abs(gap) < pooled else ''))

    # Validation TIES. vel_p8.pt and vel_p8_s4.pt both score 98.4979, and
    # taking max() over (metric, path) broke that tie by path string, so
    # "seeds/" beat "grid/" alphabetically and decided a full point of room
    # accuracy (93.42 vs 92.38) by alphabetical order. Validation has 3728
    # frames over 28 pieces and simply cannot separate these checkpoints.
    #
    # Held-out can: 40594 onsets over 80 pieces, and it is disjoint from room,
    # so using it to choose the draw as well as the config keeps room
    # untouched. Validation stays as the tie-breaker's tie-breaker.
    order = sorted(zip(summary[win]['heldout'], summary[win]['paths']),
                   key=lambda t: (-t[0], -valid_metric(t[1])))
    best = order[0][1]
    print(f'\n=== CHECKPOINT CHOSEN ON HELD-OUT: {best.split("/")[-1]} '
          f'(held-out {order[0][0]:.2f}, valid {valid_metric(best):.4f}) ===')
    print('    validation ties at 4 decimals here and cannot pick; held-out '
          'has 10x the onsets and is still disjoint from room')

    room = load_with_feat('/scratch/pmohseni/omr/candf256/room.npz')
    acc, _ = rollout(load_ckpt(best)[0], room, blend=0.7)
    ir_room, _ = rollout(load_ckpt(f'{M}/ir_only.pt')[0], room, blend=0.7)
    print(f'\n################ THE NUMBER ################')
    print(f'  model  {best}')
    print(f'  room pct@0.5s  {acc:.2f}   (ir_only {ir_room:.2f}, '
          f'cyolo_sb argmax 79.97)')
    print(f'  confirm through the harness before quoting.')


if __name__ == '__main__':
    main()
