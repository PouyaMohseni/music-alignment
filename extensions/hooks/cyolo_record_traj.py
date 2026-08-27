"""Record the full decoded TRAJECTORY, not just its per-frame error.

The aggregate pct@0.5s says how often the tracker is within half a second. It
says nothing about what the tracker is DOING: whether a failure is a brief
wobble or a commitment to the wrong repeat of a phrase, whether it recovers,
whether it jumped to the wrong staff, or whether the passage was simply dense.
Those questions need the path itself.

Per scored frame we keep the harness's own quantities:

  frame            spectrogram frame index; frame / FPS is the time in the audio
  x_pred, y_pred   predicted centre on the page, in unscaled pixels
  x_gt,   y_gt     ground truth, same coordinates
  xu_pred, xu_gt   the same x after staff assignment and unrolling
  t_pred, t_gt     both mapped through the piece's interpolator to onset frames
  staff_pred, staff_gt

|t_pred - t_gt| is exactly the frame_diff the metric reports, so a trajectory
recorded here reproduces the published number rather than approximating it.
"""
from __future__ import annotations

import os

import numpy as np

REC: dict = {}
_BATCH = {'frames': None}
_KEYS = ('frame', 'x_pred', 'y_pred', 'x_gt', 'y_gt', 'xu_pred', 'xu_gt',
         't_pred', 't_gt', 'staff_pred', 'staff_gt')


def _slot(f):
    return REC.setdefault(f, {k: [] for k in _KEYS})


def patch_traj():
    import cyolo_score_following.dataset as ds

    _orig = ds.compute_batch_stats

    def compute_batch_stats(detections, true_positions, piece_stats, file_names,
                            file_interpols, file_add_per_staff):
        out = _orig(detections, true_positions, piece_stats, file_names,
                    file_interpols, file_add_per_staff)
        gt = true_positions.float().cpu()
        pred = detections[:, :2].detach().cpu()
        frames = _BATCH['frames']
        for num, fname in enumerate(file_names):
            g = gt[((gt[:, 0] == num) & (gt[:, 1] == 0))]
            if len(g) != 1:
                continue
            g = g[0, 2:4]
            staff_coords, aps = file_add_per_staff[num]
            sc = np.asarray(staff_coords, np.float64)
            ap = np.asarray(aps, np.float64)
            interp = file_interpols[num]
            # staff assignment is nearest-coordinate, exactly as the metric does it
            sp = int(np.argmin(np.abs(sc - float(pred[num][1]))))
            sg = int(np.argmin(np.abs(sc - float(g[1]))))
            xup = float(pred[num][0]) + ap[sp]
            xug = float(g[0]) + ap[sg]
            s = _slot(fname)
            for k, v in (('frame', int(frames[num]) if frames is not None else -1),
                         ('x_pred', float(pred[num][0])), ('y_pred', float(pred[num][1])),
                         ('x_gt', float(g[0])), ('y_gt', float(g[1])),
                         ('xu_pred', xup), ('xu_gt', xug),
                         ('t_pred', float(interp(xup))), ('t_gt', float(interp(xug))),
                         ('staff_pred', sp), ('staff_gt', sg)):
                s[k].append(v)
        return out

    ds.compute_batch_stats = compute_batch_stats
    ds._traj_patched = True

    _orig_it = ds.iterate_dataset

    def iterate_dataset(network, dataloader, criterion, optimizer=None, **kw):
        class _W:
            def __init__(self, dl):
                self.dl = dl

            def __len__(self):
                return len(self.dl)

            def __iter__(self):
                for data in self.dl:
                    _BATCH['frames'] = getattr(data, 'frames', None)
                    yield data

        return _orig_it(network, _W(dataloader), criterion, optimizer=optimizer, **kw)

    ds.iterate_dataset = iterate_dataset
    print('[TRAJ] trajectory recorder active', flush=True)


def dump(path):
    out = {}
    n = 0
    for fname, d in REC.items():
        n += len(d['frame'])
        for k, v in d.items():
            out[f'{fname}||{k}'] = np.asarray(
                v, np.int32 if k in ('frame', 'staff_pred', 'staff_gt') else np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print(f'[TRAJ] wrote {path}: {len(REC)} pages, {n} frames', flush=True)
