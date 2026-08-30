"""Dump the frozen detector's candidate boxes, with everything a selector needs.

WHY
---
A perfect selector over these very candidates scores 99.7 where the shipped
argmax scores 80.0 and our hand-tuned decoder scores 86.5. So the perception
model is not the constraint and the remaining 13 points are a SELECTION problem.
That is worth learning rather than hand-designing, and it is learnable on the
data we actually have: the unit becomes a (frame, candidate) pair, tens of
thousands of them per split, instead of the 353 pieces that drowned every
previous attempt to add capacity.

WHAT IS RECORDED, AND WHY IT IS RAW
-----------------------------------
Every feature a scorer would use that depends on the PREVIOUS PREDICTION is
computed offline from this dump rather than baked in here. That is deliberate:
the policy's own past positions are part of the input, so a teacher-forced pass
and a DAgger pass need different relative features over the SAME candidates. If
those were frozen at dump time, every change of rollout strategy would mean
re-running the detector over the whole training split.

Per scored frame:
  frame              spectrogram frame index (gives elapsed time between steps)
  t_gt, x_gt         ground truth in onset-frame and unrolled-pixel coordinates
  cand[xu,y,w,h,obj,t]   note candidates, sorted by objectness (rank 0 first)
  bar[...], sys[...] the best bar and system box for the frame

Coordinates match compute_batch_stats exactly: scaled by scale_factor, staff
assigned by nearest staff_coord, x unrolled by add_per_staff, then mapped
through the piece's own interpolator. The labels are therefore the harness's
own notion of error, not a reimplementation of it.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from extensions.hooks.cyolo_z_capture import LAST_Z as _Z

# 256 because the decode sweep saturates there: fewer would mean the scorer is
# trained on a candidate set the inference path does not reproduce.
MAXK = int(os.environ.get('DUMP_MAXK', '256'))
# features are 128 floats per candidate, so they dominate the dump: 128 of them
# per frame is 5.4 GB over the training split, 256 would be 11. The oracle-best
# candidate sits at rank p90=27, so 128 loses essentially nothing.
FEATK = int(os.environ.get('DUMP_FEATK', '0'))

REC: dict = {}
_CAND: dict = {}
_BATCH = {'scale_factors': None, 'frames': None}

_FIELDS = ('frame', 't_gt', 'x_gt', 'lens', 'ntot', 'cand', 'bar', 'sys', 'z', 'feat')


def _slot(fname):
    return REC.setdefault(fname, {k: [] for k in _FIELDS})


def _best_of_class(x, cls, sf):
    """Highest-objectness box of `cls`, scaled. Zeros (with obj=0) if absent."""
    rows = x[x[:, -1] == cls]
    if rows.shape[0] == 0:
        return np.zeros(5, np.float32)
    b = rows[int(rows[:, 4].argmax())].detach().cpu().numpy()
    return np.array([b[0] * sf, b[1] * sf, b[2] * sf, b[3] * sf, b[4]], np.float32)


def patch_dump():
    import cyolo_score_following.dataset as ds
    import cyolo_score_following.utils.general as gen

    _prev = ds.get_max_box

    def get_max_box(prediction, class_id=0):
        if class_id == 0:
            _CAND.clear()
            sfs = _BATCH['scale_factors']
            for xi, x in enumerate(prediction):
                sf = float(sfs[xi]) if sfs is not None else 1.0
                m0 = x[:, -1] == 0
                sel = x[m0]
                if sel.shape[0] == 0:
                    _CAND[xi] = None
                    continue
                # class 0 candidates are exactly the P3 block and it comes
                # first in the concatenation, so these indices address the
                # captured feature map directly
                idx0 = torch.nonzero(m0).squeeze(-1).cpu().numpy()
                obj = sel[:, 4].detach().cpu().numpy().astype(np.float32)
                keep = (np.argpartition(-obj, MAXK)[:MAXK] if obj.shape[0] > MAXK
                        else np.arange(obj.shape[0]))
                order = keep[np.argsort(-obj[keep])]
                box = sel[:, :4].detach().cpu().numpy()[order] * sf
                # ntot is the count BEFORE capping, so crowding stays honest
                # whatever cap either side happens to apply
                ft = None
                if FEATK:
                    from extensions.hooks.cyolo_feat_capture import gather
                    ft = gather(xi, idx0[order][:FEATK])
                _CAND[xi] = (box, obj[order], int(obj.shape[0]),
                             _best_of_class(x, 1, sf), _best_of_class(x, 2, sf), ft)
        return _prev(prediction, class_id=class_id)

    ds.get_max_box = get_max_box
    gen.get_max_box = get_max_box

    _orig_cbs = ds.compute_batch_stats

    def compute_batch_stats(detections, true_positions, piece_stats, file_names,
                            file_interpols, file_add_per_staff):
        out = _orig_cbs(detections, true_positions, piece_stats, file_names,
                        file_interpols, file_add_per_staff)
        gt = true_positions.float().cpu()
        frames = _BATCH['frames']
        for num, fname in enumerate(file_names):
            gt_note = gt[((gt[:, 0] == num) & (gt[:, 1] == 0))]
            if len(gt_note) != 1:
                continue
            gt_note = gt_note[0, 2:4]
            staff_coords, add_per_staff = file_add_per_staff[num]
            sc = np.asarray(staff_coords, np.float32)
            aps = np.asarray(add_per_staff, np.float32)
            interp = file_interpols[num]

            def unroll(x, y):
                return x + aps[np.argmin(np.abs(sc[None, :] - np.atleast_1d(y)[:, None]), 1)]

            x_gt = float(unroll(np.array([float(gt_note[0])]),
                                np.array([float(gt_note[1])]))[0])
            s = _slot(fname)
            zz = _Z.get('z')
            s['z'].append(zz[num] if zz is not None and num < len(zz)
                          else np.zeros(128, np.float32))
            s['frame'].append(int(frames[num]) if frames is not None else -1)
            s['t_gt'].append(float(interp(x_gt)))
            s['x_gt'].append(x_gt)

            cand = _CAND.get(num)
            if cand is None:
                s['lens'].append(0)
                s['ntot'].append(0)
                if FEATK:
                    s['feat'].append(np.zeros((0, 1), np.float16))
                s['cand'].append(np.zeros((0, 6), np.float32))
                s['bar'].append(np.zeros(5, np.float32))
                s['sys'].append(np.zeros(5, np.float32))
                continue
            box, obj, ntot, bar, sysb, ft = cand
            xu = unroll(box[:, 0], box[:, 1])
            t = np.asarray(interp(xu), dtype=np.float32).ravel()
            s['lens'].append(len(xu))
            s['ntot'].append(ntot)
            if FEATK:
                s['feat'].append(ft if ft is not None
                                 else np.zeros((0, 1), np.float16))
            s['cand'].append(np.stack([xu, box[:, 1], box[:, 2], box[:, 3],
                                       obj, t], 1).astype(np.float32))
            # bar/system centres are unrolled too, so distances live in the same
            # coordinate as the candidates and the ground truth
            bar[0] = unroll(bar[:1], bar[1:2])[0]
            sysb[0] = unroll(sysb[:1], sysb[1:2])[0]
            s['bar'].append(bar)
            s['sys'].append(sysb)
        return out

    ds.compute_batch_stats = compute_batch_stats
    ds._dump_patched = True

    _orig_iterate = ds.iterate_dataset

    def iterate_dataset(network, dataloader, criterion, optimizer=None, **kwargs):
        class _W:
            def __init__(self, dl):
                self.dl = dl

            def __len__(self):
                return len(self.dl)

            def __iter__(self):
                for data in self.dl:
                    _BATCH['scale_factors'] = data.scale_factors
                    _BATCH['frames'] = getattr(data, 'frames', None)
                    yield data

        return _orig_iterate(network, _W(dataloader), criterion,
                             optimizer=optimizer, **kwargs)

    ds.iterate_dataset = iterate_dataset
    print(f'[DUMP] candidate dumper active (MAXK={MAXK})', flush=True)


def dump(path):
    out = {}
    nf = 0
    for fname, d in REC.items():
        n = len(d['frame'])
        nf += n
        out[f'{fname}||frame'] = np.asarray(d['frame'], np.int32)
        out[f'{fname}||t_gt'] = np.asarray(d['t_gt'], np.float32)
        out[f'{fname}||x_gt'] = np.asarray(d['x_gt'], np.float32)
        out[f'{fname}||lens'] = np.asarray(d['lens'], np.int32)
        out[f'{fname}||ntot'] = np.asarray(d['ntot'], np.int32)
        out[f'{fname}||cand'] = (np.concatenate(d['cand']) if n
                                 else np.zeros((0, 6), np.float32))
        out[f'{fname}||bar'] = (np.stack(d['bar']) if n else np.zeros((0, 5), np.float32))
        out[f'{fname}||sys'] = (np.stack(d['sys']) if n else np.zeros((0, 5), np.float32))
        out[f'{fname}||z'] = (np.stack(d['z']) if n else np.zeros((0, 128), np.float32))
        if FEATK and d['feat']:
            out[f'{fname}||flens'] = np.array([a.shape[0] for a in d['feat']], np.int32)
            out[f'{fname}||feat'] = np.concatenate(d['feat']) if n else np.zeros((0, 1), np.float16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print(f'[DUMP] wrote {path}: {len(REC)} pieces, {nf} frames', flush=True)
    REC.clear()
