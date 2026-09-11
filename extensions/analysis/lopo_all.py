"""Every trained configuration under ONE fixed protocol, selected on validation.

The protocol is fixed before any model is scored and is the one from
lopo_tuning.py: for each of the 16 real pieces the decoder's scalars are chosen
on the other 15 with pieces weighted equally, the held-out piece is decoded with
them, and accuracy is pooled over onsets as the published metric requires. Only
the MODEL varies here.

Three columns per checkpoint:

  valid     the greedy rollout on the MSMD validation split, recorded when the
            checkpoint was written; never sees room or the held-out pieces
  held-out  80 unseen pieces at 12 dB SNR, one rollout at blend 0.7
  room      the leave-one-piece-out number, i.e. what this model would score

Selection uses `valid` or `held-out`. The room column is read once, for the
winner, and reporting the best room entry instead would be selecting on the
test set -- the table prints it for every model only so the spread is visible.

A --residual checkpoint has no scalars to select: its prior is baked into the
training objective and decoding is argmax(s + h), the blend formula at 0.5. Its
room column is therefore a plain evaluation, not a cross-validated one, which
makes it the cleanest entry in the table if it is competitive.
"""
from __future__ import annotations

import argparse
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
ROOM = '/scratch/pmohseni/omr/candf256/room.npz'
HV = '/scratch/pmohseni/omr/candhv256/valid_snr12.npz'
BLENDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID = list(itertools.product((3.0, 4.5, 6.0, 8.0, 10.0), (12.0, 18.0, 27.0),
                              (-4.0, -6.0, -8.0)))
RESID_PRIOR = (6.0, 18.0, -6.0)      # baked into the residual training objective

CONFIGS = [
    ('featureless', f'{M}/ir_only.pt'),
    ('vel_p8', f'{M}/grid/vel_p8.pt'),
    ('vel_p8', f'{M}/seeds/vel_p8_s*.pt'),
    ('vel_p8_f256', f'{M}/f256/vel_p8_f256*.pt'),
    ('nbr proj8', f'{M}/nbr/nbr_s*.pt'),
    ('nbr proj32', f'{M}/nbr/nbrp32_s*.pt'),
    ('nbr proj64', f'{M}/nbr/nbrp64_s*.pt'),
    ('tempo', f'{M}/tempo/tempo_s*.pt'),
    ('pitch', f'{M}/pitchsc/pitchsc_s*.pt'),
    ('dagger', f'{M}/dagger/dagger_s*.pt'),
    ('dagrec', f'{M}/dagrec/dagrec_w*_s*.pt'),
    ('crf', f'{M}/crf/crf_s*.pt'),
    ('nohist', f'{M}/extra/nohist_s*.pt'),
    ('residual', f'{M}/resid/resid_s*.pt'),
    ('cyolo detector', f'{M}/extra/cyolo_s*.pt'),
    ('cyolo_sb_a detector', f'{M}/extra/cyolo_sb_a_s*.pt'),
    ('MERT audio', f'{M}/abl/h1_s*.pt'),
    ('DINOv2 image', f'{M}/abl/dino_s*.pt'),
    ('MERT + DINOv2', f'{M}/abl/h1dino_s*.pt'),
]


def is_residual(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return bool((d.get('extra') or {}).get('residual', False))


def valid_of(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return float((d.get('extra') or {}).get('valid_rollout', float('nan')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--skip_heldout', action='store_true')
    a = ap.parse_args()

    room = load_with_feat(ROOM)
    arg, pg = rollout_argmax(room)
    pieces = piece_of(pg)
    names = sorted(set(pieces))

    def macro_on_train(hits, held):
        return np.mean([hits[pieces == q].mean() for q in names if q != held])

    hand = {}
    for pr in GRID:
        h, _ = rollout_hand(room, fwd=pr[0], sigma=pr[1], jump=pr[2])
        hand[pr] = h
    fold_prior = {p: max(GRID, key=lambda k: macro_on_train(hand[k], p)) for p in names}
    lo_hand = np.zeros(len(pieces), bool)
    for p in names:
        lo_hand[pieces == p] = hand[fold_prior[p]][pieces == p]
    print(f'{len(arg)} onsets / {len(names)} pieces   argmax {100 * arg.mean():.2f}   '
          f'prior (cross-validated) {100 * lo_hand.mean():.2f}\n', flush=True)

    todo = []
    for cfg, pat in CONFIGS:
        for p in sorted(glob.glob(pat)):
            todo.append((cfg, p))
    todo = todo[a.shard::a.num_shards]
    print(f'{len(todo)} checkpoints in this shard\n', flush=True)

    hv = None if a.skip_heldout else load_with_feat(HV)
    print(f'{"configuration":22s} {"checkpoint":22s} {"valid":>6s} {"held-out":>8s} '
          f'{"room":>6s}  scalars')
    for cfg, path in todo:
        try:
            model = load_ckpt(path)[0]
        except Exception as e:
            print(f'{cfg:22s} {os.path.basename(path):22s}  load failed: {e}', flush=True)
            continue
        resid = is_residual(path)
        hvs = float('nan')
        if hv is not None:
            try:
                h, _ = rollout_hits(model, hv, blend=(0.5 if resid else 0.7))
                hvs = 100.0 * h.mean()
            except Exception:
                pass
        if resid:
            hits, _ = rollout_hits(model, room, blend=0.5, fwd=RESID_PRIOR[0],
                                   sigma=RESID_PRIOR[1], jump=RESID_PRIOR[2])
            lo, note = 100.0 * hits.mean(), 'none (fixed by training)'
        else:
            full = {}
            for pr in sorted(set(fold_prior.values())):
                for b in BLENDS:
                    s, _ = rollout_hits(model, room, blend=b, fwd=pr[0],
                                        sigma=pr[1], jump=pr[2])
                    full[(pr, b)] = s
            picked = np.zeros(len(pieces), bool)
            betas = []
            for p in names:
                pr = fold_prior[p]
                b = max(BLENDS, key=lambda v: macro_on_train(full[(pr, v)], p))
                betas.append(b)
                picked[pieces == p] = full[(pr, b)][pieces == p]
            lo = 100.0 * picked.mean()
            note = f'beta {min(betas):.1f}-{max(betas):.1f}'
        print(f'{cfg:22s} {os.path.basename(path)[:22]:22s} {valid_of(path):6.2f} '
              f'{hvs:8.2f} {lo:6.2f}  {note}', flush=True)


if __name__ == '__main__':
    main()
