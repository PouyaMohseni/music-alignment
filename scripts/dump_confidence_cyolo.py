"""Dump per-frame confidence signals + tracking error for a released CYOLO checkpoint.

WHY
---
The score-following literature has no calibration evaluation: no published system
reports whether its own confidence signal predicts when it is lost.  The two
existing "am I lost?" mechanisms are hand-built binary rules:

  * Brazier & Widmer, EUSIPCO 2021 -- "reliability factor" rf in {0,1}; rf = 1 iff
    the last 30 tracked score positions fit a line with slope in [0.5, 1.5].
  * CODA (ISMIR 2026) -- silence-driven "break mode".

To score those against a calibrated continuous signal we need, per frame:
    (a) the tracking error in seconds, and
    (b) every candidate confidence signal, on the SAME frames.
This script produces exactly that for the CYOLO family (Henkel & Widmer 2021,
Frontiers in CS), whose released checkpoints are the current published SOTA on
MSMD-Rec.

WHAT IS DUMPED (one record per evaluated frame, grouped by page)
----------------------------------------------------------------
    err_frames   |interpol_c2o(x_pred) - interpol_c2o(x_gt)|, in *onset frames*.
                 Identical definition to cyolo_score_following.dataset
                 .compute_batch_stats, which is what the paper's
                 "tracked frame ratio" table is built from.  Seconds = /FPS,
                 FPS = 22050/1102 = 20.009.
    is_onset     True iff this frame carries a note onset.  The published
                 pct@0.5s metric is computed over is_onset frames only
                 (eval.py --only_onsets), so all headline AUROCs are reported
                 on this subset.
    conf1        max objectness over all class-0 ("Note") anchors, post-sigmoid.
                 This is the score get_max_box() argmaxes over, i.e. the
                 system's own internal notion of how sure it is.
    conf2        second-highest objectness among class-0 anchors whose x-bin is
                 at least MIN_SEP bins away from the argmax bin -- a *spatially
                 separated* runner-up, so conf1-conf2 measures multimodality of
                 the position posterior rather than neighbouring-anchor noise.
    ent_x        Shannon entropy (nats) of the normalised x-marginal of the
                 objectness field: p(b) = max objectness in x-bin b, normalised
                 to sum 1 over N_BINS bins spanning the scaled score width.
    x_pred,x_gt  unrolled x pixel of the prediction / ground truth, original
                 (unscaled) pixel units -- needed for the Brazier & Widmer
                 slope rule, which regresses tracked position against time.
    frame        audio frame index, so per-frame audio RMS (silence baseline)
                 can be joined on it afterwards.

PROVENANCE
----------
Model + data code: /scratch/pmohseni/datasets/cyolo_score_following (unmodified,
imported not copied).  Checkpoints: trained_models/{cyolo,cyolo_sb,cyolo_sb_a}.
Data: /scratch/pmohseni/datasets/cyolo_data/msmd/msmd_rp with split_files/
{room,do,rp_synth}_split.yaml -- the same 25 pages / 16 pieces / 4415 onsets that
results/verify_cyolo_bar-143056.log scores.

USAGE
-----
    python scripts/dump_confidence_cyolo.py --model cyolo_sb --tier room \
        --out results/calibration/cyolo_sb_room.npz [--only_onsets]

Without --only_onsets every audio frame is evaluated (~22.4k frames for room),
which the Brazier & Widmer rule needs: it is defined over consecutive tracked
positions, not over the sparse onset subset.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

CY = '/scratch/pmohseni/datasets/cyolo_score_following'
if CY not in sys.path:
    sys.path.insert(0, CY)

from cyolo_score_following.dataset import load_dataset, CustomBatch  # noqa: E402
from cyolo_score_following.models.yolo import load_pretrained_model  # noqa: E402
from cyolo_score_following.utils.data_utils import FPS, HOP_SIZE, FRAME_SIZE  # noqa: E402

N_BINS = 64      # x-marginal resolution for the entropy signal
MIN_SEP = 3      # bins the runner-up mode must be away from the argmax bin


class ConfBatch(CustomBatch):
    """CustomBatch drops is_onset and t; we need both to join errors to frames."""

    def __init__(self, batch):
        super().__init__(batch)
        self.is_onset = [bool(x['is_onset']) for x in batch]
        self.t = [int(x['t']) for x in batch]


def conf_collate(batch):
    return ConfBatch(batch)


def x_marginal(xs, confs, width, n_bins=N_BINS):
    """max-pool objectness into n_bins x-bins, normalise to a distribution."""
    idx = np.clip((xs / max(width, 1e-6) * n_bins).astype(np.int64), 0, n_bins - 1)
    p = np.zeros(n_bins, dtype=np.float64)
    np.maximum.at(p, idx, confs)
    s = p.sum()
    if s <= 0:
        return p, idx
    return p / s, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='cyolo_sb')
    ap.add_argument('--tier', default='room', choices=['room', 'do', 'rp_synth'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--scale_width', type=float, default=416)
    ap.add_argument('--only_onsets', action='store_true')
    args = ap.parse_args()

    data = '/scratch/pmohseni/datasets/cyolo_data/msmd'
    ckpt = f'{CY}/trained_models/{args.model}/best_model.pt'
    split = f'{data}/split_files/{args.tier}_split.yaml'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network, _ = load_pretrained_model(ckpt)
    predict_sb = network.nc == 3
    network.to(device).eval()
    print(f'model={args.model} tier={args.tier} nc={network.nc} device={device}', flush=True)

    dataset = load_dataset([f'{data}/msmd_rp'], augment=False, scale_width=args.scale_width,
                           split_files=[split], only_onsets=args.only_onsets,
                           load_audio=False, predict_sb=predict_sb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=conf_collate)

    rec = {}
    for bi, d in enumerate(loader):
        scores = d.scores.to(device)
        scale_factors = d.scale_factors.to(device)
        perf = [p.to(device) for p in d.perf]

        with torch.no_grad():
            inference_out, _ = network(score=scores, perf=perf)
        inference_out = inference_out.detach().cpu()

        unscaled_targets = d.unscaled_targets.float().cpu()
        sf = d.scale_factors.cpu().numpy()[:, 0]
        width = scores.shape[-1]

        for n, fname in enumerate(d.file_names):
            out = inference_out[n]
            m = out[:, -1] == 0                     # class 0 == "Note" detection layer
            out = out[m].numpy()
            xs, ys, confs = out[:, 0], out[:, 1], out[:, 4]

            k = int(np.argmax(confs))
            conf1 = float(confs[k])
            p, binidx = x_marginal(xs, confs, width)
            ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            far = np.abs(binidx - binidx[k]) >= MIN_SEP
            conf2 = float(confs[far].max()) if far.any() else 0.0

            x_pred_s, y_pred_s = xs[k] * sf[n], ys[k] * sf[n]

            gt = unscaled_targets[(unscaled_targets[:, 0] == n) & (unscaled_targets[:, 1] == 0)]
            if len(gt) != 1:
                continue
            x_gt_s, y_gt_s = float(gt[0, 2]), float(gt[0, 3])

            staff_coords, add_per_staff = d.add_per_staff[n]
            sid_p = int(np.argwhere(min(staff_coords, key=lambda y: abs(y - y_pred_s)) == staff_coords).item())
            sid_g = int(np.argwhere(min(staff_coords, key=lambda y: abs(y - y_gt_s)) == staff_coords).item())
            xu_p = x_pred_s + add_per_staff[sid_p]
            xu_g = x_gt_s + add_per_staff[sid_g]
            err = float(abs(d.interpols[n](xu_p) - d.interpols[n](xu_g)))

            r = rec.setdefault(fname, {k2: [] for k2 in
                                       ('frame', 'is_onset', 'err_frames', 'conf1', 'conf2',
                                        'ent_x', 'x_pred', 'x_gt')})
            r['frame'].append((d.t[n] - FRAME_SIZE) // HOP_SIZE)
            r['is_onset'].append(d.is_onset[n])
            r['err_frames'].append(err)
            r['conf1'].append(conf1)
            r['conf2'].append(conf2)
            r['ent_x'].append(ent)
            r['x_pred'].append(float(xu_p))
            r['x_gt'].append(float(xu_g))

        if bi % 25 == 0:
            print(f'batch {bi}/{len(loader)}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    flat = {}
    for fname, r in rec.items():
        order = np.argsort(np.asarray(r['frame']))   # restore temporal order
        for k2, v in r.items():
            flat[f'{fname}||{k2}'] = np.asarray(v)[order]
    flat['__fps__'] = np.asarray([FPS])
    flat['__meta__'] = np.asarray([args.model, args.tier, str(args.only_onsets)])
    np.savez_compressed(args.out, **flat)

    n = sum(len(r['err_frames']) for r in rec.values())
    n_on = sum(int(np.sum(r['is_onset'])) for r in rec.values())
    errs = np.concatenate([np.asarray(r['err_frames'])[np.asarray(r['is_onset'], bool)] for r in rec.values()])
    print(f'wrote {args.out}: {len(rec)} pages, {n} frames, {n_on} onset frames')
    print(f'sanity pct@0.5s over onsets = {100 * np.mean(errs / FPS <= 0.5):.1f}')


if __name__ == '__main__':
    main()
