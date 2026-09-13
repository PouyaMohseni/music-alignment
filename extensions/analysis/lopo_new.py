"""The configurations trained after lopo_all ran, each against ITS OWN dump.

lopo_all.py scored every checkpoint on cyolo_sb's candidates, which is only
correct for scorers trained on them. A scorer fitted to another detector's
boxes, or to DINOv2 features, has to be read from that detector's dump or the
features it sees at test time are not the ones it was fitted to.

Protocol is unchanged: the prior and blend are chosen per fold on the other
fifteen pieces with pieces weighted equally, accuracy pooled over onsets. A
residual checkpoint has nothing to choose -- its prior is inside the training
objective and decoding is argmax(s + h) -- so its row is a plain evaluation.
"""
from __future__ import annotations

import glob
import itertools
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import piece_of, rollout_hits
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand
from extensions.heads.cand_scorer import load as load_ckpt

M = '/scratch/pmohseni/omr/scorer'
O = '/scratch/pmohseni/omr'
BLENDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID = list(itertools.product((3.0, 4.5, 6.0, 8.0, 10.0), (12.0, 18.0, 27.0),
                              (-4.0, -6.0, -8.0)))
# (label, checkpoint glob, room dump, fixed prior or None)
JOBS = [
    ('residual (clean prior)', f'{M}/resid/resid_s*.pt', f'{O}/candf256/room.npz', (10.0, 18.0, -8.0)),
    ('residual (old prior)',   f'{M}/resid/residold_s*.pt', f'{O}/candf256/room.npz', (6.0, 18.0, -6.0)),
    ('history-free',           f'{M}/extra/nohist_s*.pt', f'{O}/candf256/room.npz', None),
    ('detector: CYOLO',        f'{M}/extra/cyolo_s[0-9].pt', f'{O}/cand_cyolo/room.npz', None),
    ('detector: CYOLO-SB+A',   f'{M}/extra/cyolo_sb_a_s*.pt', f'{O}/cand_cyolo_sb_a/room.npz', None),
    # the same recipe minus --use_feat, on each detector's own candidates: the
    # 128 backbone numbers are worth +3.4 on synthetic validation and nothing
    # on room for cyolo_sb, and 94.38 was trained WITH them, so their
    # contribution on the strongest detector was unmeasured until these ran
    ('CYOLO, no image feats',  f'{M}/extra/cyolo_nofeat_s*.pt', f'{O}/cand_cyolo/room.npz', None),
    ('CYOLO-SB+A, no image feats', f'{M}/extra/cyolo_sb_a_nofeat_s*.pt', f'{O}/cand_cyolo_sb_a/room.npz', None),
    ('image: DINOv2',          f'{M}/abl/dino_s*.pt', f'{O}/candf_dino/room.npz', None),
    ('audio: MERT',            f'{M}/abl/h1_s*.pt', f'{O}/candh1/room.npz', None),
    ('audio+image swap',       f'{M}/abl/h1dino_s*.pt', f'{O}/candh1_dino/room.npz', None),
    # the five retrained encoder arms: three audio (LSTM control, CODA's Mamba
    # tower, CNN keeping only the Mamba recurrence) and two image (DINOv2 stem
    # against its own control, both without the page shift)
    ('encoder: LSTM control',  f'{M}/enc/lstm_s*.pt',     f'{O}/cand_enc_lstm/room.npz', None),
    ('encoder: Mamba tower',   f'{M}/enc/mamba_s*.pt',    f'{O}/cand_enc_mamba/room.npz', None),
    ('encoder: CNN+Mamba',     f'{M}/enc/cnnmamba_s*.pt', f'{O}/cand_enc_cnnmamba/room.npz', None),
    ('encoder: DINOv2 stem',   f'{M}/enc/dinov2_s*.pt',   f'{O}/cand_enc_dinov2/room.npz', None),
    ('encoder: CNN control',   f'{M}/enc/cnn_s*.pt',      f'{O}/cand_enc_cnn/room.npz', None),
]


def main():
    cache = {}
    print(f'{"configuration":24s} {"n":>2s} {"argmax":>7s} {"prior":>7s} {"room":>7s} {"sd":>5s}  notes')
    for label, pat, dump, fixed in JOBS:
        paths = sorted(glob.glob(pat))
        if not paths or not os.path.exists(dump):
            print(f'{label:24s}  missing ({len(paths)} ckpt, dump {"ok" if os.path.exists(dump) else "absent"})')
            continue
        if dump not in cache:
            pages = load_with_feat(dump)
            arg, pg = rollout_argmax(pages)
            pieces = piece_of(pg)
            names = sorted(set(pieces))
            hand = {}
            for pr in GRID:
                h, _ = rollout_hand(pages, fwd=pr[0], sigma=pr[1], jump=pr[2])
                hand[pr] = h
            fp = {p: max(GRID, key=lambda k: np.mean(
                [hand[k][pieces == q].mean() for q in names if q != p])) for p in names}
            lo_h = np.zeros(len(pieces), bool)
            for p in names:
                lo_h[pieces == p] = hand[fp[p]][pieces == p]
            cache[dump] = (pages, arg, pieces, names, fp, lo_h)
        pages, arg, pieces, names, fold_prior, lo_hand = cache[dump]

        accs, notes = [], ''
        for path in paths:
            model = load_ckpt(path)[0]
            if fixed is not None:
                hits, _ = rollout_hits(model, pages, blend=0.5, fwd=fixed[0],
                                       sigma=fixed[1], jump=fixed[2])
                accs.append(100.0 * hits.mean())
                notes = 'no scalars tuned'
            else:
                full = {}
                for pr in sorted(set(fold_prior.values())):
                    for b in BLENDS:
                        s, _ = rollout_hits(model, pages, blend=b, fwd=pr[0],
                                            sigma=pr[1], jump=pr[2])
                        full[(pr, b)] = s
                picked = np.zeros(len(pieces), bool)
                for p in names:
                    pr = fold_prior[p]
                    b = max(BLENDS, key=lambda v: np.mean(
                        [full[(pr, v)][pieces == q].mean() for q in names if q != p]))
                    picked[pieces == p] = full[(pr, b)][pieces == p]
                accs.append(100.0 * picked.mean())
                notes = 'cross-validated'
        a = np.array(accs)
        print(f'{label:24s} {len(a):2d} {100 * arg.mean():7.2f} {100 * lo_hand.mean():7.2f} '
              f'{a.mean():7.2f} {a.std(ddof=1) if len(a) > 1 else 0:5.2f}  {notes}', flush=True)


if __name__ == '__main__':
    main()
