"""Capture per-frame eval detail that the harness averages away.

WHY THIS EXISTS
---------------
`eval.py --save_tag` saves only `frame_diff` per piece plus SCALAR bar/system
accuracies -- dataset.py:407 reduces the per-frame 0/1 correctness list to a
mean before it is stored. Two questions need the unreduced arrays:

  1. Is C2's +3.5 on pct@0.5s consistent across pieces, or carried by two or
     three of the sixteen? A paired bootstrap needs per-piece frame_diff, which
     save_tag does give -- but it needs both arms recorded identically.

  2. WHERE do C2's bar errors land? Bar accuracy falls 0.829 -> 0.689 at the
     setting we quote as best, monotonically in lam. If those misses sit on
     frames the tracker also times badly, the prior is dragging predictions
     off-target. If they sit on frames that are timed WELL, then the box is
     temporally right and bar-wrong -- a repeat, i.e. the ambiguity we expect,
     and the tracking gain is real.

Answering (2) needs bar correctness aligned FRAME BY FRAME with frame_diff, and
that alignment is exactly what the mean destroys.

HOW
---
`compute_batch_stats` and `eval_class` both append to `piece_stats` in call
order, batch by batch, keyed by file name. We wrap both, note the list length
before delegating, and copy whatever the original appended. That keeps the two
streams index-aligned without reimplementing either metric -- the numbers stay
the harness's own.
"""
from __future__ import annotations

import os

import numpy as np

REC = {}          # fname -> {'frame_diff': [...], 'cls_1': [...], 'cls_2': [...]}


def _slot(fname, key):
    return REC.setdefault(fname, {}).setdefault(key, [])


def patch_recorder():
    import cyolo_score_following.dataset as ds

    _orig_cbs = ds.compute_batch_stats
    _orig_ec = ds.eval_class

    def compute_batch_stats(detections, true_positions, piece_stats, file_names,
                            file_interpols, file_add_per_staff):
        before = {f: len(piece_stats.get(f, {}).get('frame_diff', []))
                  for f in set(file_names)}
        out = _orig_cbs(detections, true_positions, piece_stats, file_names,
                        file_interpols, file_add_per_staff)
        for f in set(file_names):
            new = out.get(f, {}).get('frame_diff', [])[before[f]:]
            _slot(f, 'frame_diff').extend(float(v) for v in new)
        return out

    def eval_class(prediction, targets, piece_stats, file_names, scale_factors,
                   class_id=1, th=0.8):
        before = {f: len(piece_stats.get(f, {}).get(class_id, []))
                  for f in set(file_names)}
        out, boxes = _orig_ec(prediction, targets, piece_stats, file_names,
                              scale_factors, class_id=class_id, th=th)
        for f in set(file_names):
            new = out.get(f, {}).get(class_id, [])[before[f]:]
            _slot(f, f'cls_{class_id}').extend(int(v) for v in new)
        return out, boxes

    ds.compute_batch_stats = compute_batch_stats
    ds.eval_class = eval_class
    ds._recorder_patched = True
    print('[REC] per-frame recorder active', flush=True)


def dump(path):
    out = {}
    for fname, d in REC.items():
        for k, v in d.items():
            out[f'{fname}||{k}'] = np.asarray(v)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    n = len(REC)
    tot = sum(len(d.get('frame_diff', [])) for d in REC.values())
    print(f'[REC] wrote {path}: {n} pieces, {tot} frames', flush=True)
