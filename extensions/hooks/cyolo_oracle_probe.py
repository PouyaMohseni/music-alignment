"""How much is left in the detector's own candidate list?

WHY THIS EXISTS
---------------
The temporal decode took 79.9 -> 85.9 without touching a single weight. It works
by re-ranking candidates the frozen detector already emits: on the frames it
fixes, the right box was ALREADY THERE, just not first. That fact bounds every
post-hoc method we can still build, and the bound is measurable.

For each frame, over every class-0 candidate, compute the harness's own
frame_diff. Then:

  top1      -- pick the highest-objectness candidate      (~= raw cyolo_sb)
  actual    -- whatever the installed decoder picked       (= our 85.9)
  oracle    -- pick the candidate closest to ground truth  (THE CEILING)

`oracle` is the score of a perfect re-ranker over a frozen backbone. If it sits
near 100, the backbone already sees the answer and the remaining error is a
SELECTION problem -- so parameters belong in a scorer over candidates, trained on
frames (millions) rather than pieces (354), which is the sample-size regime every
previous capacity-add died in. If oracle sits near 87, selection is nearly
exhausted and only the features themselves can move, which is the expensive road.

We also record the objectness RANK of the oracle-best candidate. Rank tells us
what a scorer would have to do: rank 2-5 is a re-ranking problem, rank 400 means
objectness carries almost no signal about which candidate is right.

HOW
---
`get_max_box(prediction, class_id=0)` is called immediately before
`compute_batch_stats` on the same batch, so we stash the raw candidates there and
consume them in the stats wrapper, where the interpolator and add_per_staff
needed to convert x -> onset frame actually live. Numbers stay the harness's own:
we call its interpolator, its unrolling, its staff assignment.
"""
from __future__ import annotations

import os

import numpy as np

MAXK = int(os.environ.get('ORACLE_MAXK', '128'))

# fname -> {'t_gt': [...], 't_cand': [array], 'obj': [array], 'fd_actual': [...]}
REC: dict = {}
_CAND: dict = {}
_BATCH = {'scale_factors': None}


def _slot(fname):
    return REC.setdefault(fname, {'t_gt': [], 'fd_actual': [], 't_cand': [], 'obj': []})


def patch_oracle():
    import cyolo_score_following.dataset as ds
    import cyolo_score_following.utils.general as gen

    _prev = ds.get_max_box

    def get_max_box(prediction, class_id=0):
        if class_id == 0:
            _CAND.clear()
            sfs = _BATCH['scale_factors']
            for xi, x in enumerate(prediction):
                sel = x[x[:, -1] == 0]
                if sel.shape[0] == 0:
                    _CAND[xi] = None
                    continue
                obj = sel[:, 4].detach().cpu().numpy().astype(np.float32)
                # keep the top-MAXK by objectness: 256 already saturates the
                # decode sweep, so this discards nothing the decoder could use
                if obj.shape[0] > MAXK:
                    keep = np.argpartition(-obj, MAXK)[:MAXK]
                else:
                    keep = np.arange(obj.shape[0])
                order = keep[np.argsort(-obj[keep])]      # rank 0 = most confident
                sf = float(sfs[xi]) if sfs is not None else 1.0
                _CAND[xi] = (sel[:, :2].detach().cpu().numpy()[order] * sf,
                             obj[order])
        return _prev(prediction, class_id=class_id)

    ds.get_max_box = get_max_box
    gen.get_max_box = get_max_box

    _orig_cbs = ds.compute_batch_stats

    def compute_batch_stats(detections, true_positions, piece_stats, file_names,
                            file_interpols, file_add_per_staff):
        before = {f: len(piece_stats.get(f, {}).get('frame_diff', []))
                  for f in set(file_names)}
        out = _orig_cbs(detections, true_positions, piece_stats, file_names,
                        file_interpols, file_add_per_staff)

        gt = true_positions.float().cpu()
        for num, fname in enumerate(file_names):
            new = out.get(fname, {}).get('frame_diff', [])[before[fname]:]
            if len(new) != 1:
                # the harness only scores frames carrying exactly one note GT;
                # skip anything it skipped so our arrays stay index-aligned
                continue
            gt_note = gt[((gt[:, 0] == num) & (gt[:, 1] == 0))]
            if len(gt_note) != 1:
                continue
            gt_note = gt_note[0, 2:4]
            staff_coords, add_per_staff = file_add_per_staff[num]
            staff_coords = np.asarray(staff_coords)
            interp = file_interpols[num]

            sid_gt = int(np.argmin(np.abs(staff_coords - float(gt_note[1]))))
            t_gt = float(interp(float(gt_note[0]) + add_per_staff[sid_gt]))

            cand = _CAND.get(num)
            if cand is None:
                t_c = np.zeros(0, np.float32)
                obj = np.zeros(0, np.float32)
            else:
                xy, obj = cand
                sid = np.argmin(np.abs(staff_coords[None, :] - xy[:, 1:2]), axis=1)
                xs = xy[:, 0] + np.asarray(add_per_staff)[sid]
                t_c = np.asarray(interp(xs), dtype=np.float32).ravel()

            s = _slot(fname)
            s['t_gt'].append(t_gt)
            s['fd_actual'].append(float(new[0]))
            s['t_cand'].append(t_c.astype(np.float32))
            s['obj'].append(np.asarray(obj, np.float32))
        return out

    ds.compute_batch_stats = compute_batch_stats
    ds._oracle_patched = True

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
                    yield data

        return _orig_iterate(network, _W(dataloader), criterion,
                             optimizer=optimizer, **kwargs)

    ds.iterate_dataset = iterate_dataset
    print(f'[ORACLE] candidate probe active (MAXK={MAXK})', flush=True)


def summarize(th_frames=10.0):
    """th_frames=10 at FPS=20 is the 0.5s threshold the paper reports."""
    n = top1 = act = orc = 0
    ranks = []
    for d in REC.values():
        for t_gt, fd, t_c, obj in zip(d['t_gt'], d['fd_actual'], d['t_cand'], d['obj']):
            n += 1
            act += fd <= th_frames
            if t_c.size == 0:
                continue
            err = np.abs(t_c - t_gt)
            top1 += err[0] <= th_frames          # obj is sorted desc, so 0 = argmax
            b = int(np.argmin(err))
            orc += err[b] <= th_frames
            ranks.append(b)
    if n == 0:
        return {}
    ranks = np.asarray(ranks)
    return {'frames': n,
            'top1': 100.0 * top1 / n,
            'actual': 100.0 * act / n,
            'oracle': 100.0 * orc / n,
            'rank_median': float(np.median(ranks)) if ranks.size else -1,
            'rank_p90': float(np.percentile(ranks, 90)) if ranks.size else -1,
            'rank_is_0': 100.0 * float(np.mean(ranks == 0)) if ranks.size else -1}


def dump(path):
    s = summarize()
    if s:
        print(f"[ORACLE] frames={s['frames']}  top1={s['top1']:.1f}  "
              f"actual={s['actual']:.1f}  ORACLE={s['oracle']:.1f}", flush=True)
        print(f"[ORACLE] oracle-best rank: median={s['rank_median']:.0f} "
              f"p90={s['rank_p90']:.0f}  already-rank-0={s['rank_is_0']:.1f}%", flush=True)
    out = {}
    for fname, d in REC.items():
        out[f'{fname}||t_gt'] = np.asarray(d['t_gt'], np.float32)
        out[f'{fname}||fd_actual'] = np.asarray(d['fd_actual'], np.float32)
        # ragged candidate lists -> flat values + offsets
        lens = np.array([a.size for a in d['t_cand']], np.int32)
        out[f'{fname}||lens'] = lens
        out[f'{fname}||t_cand'] = (np.concatenate(d['t_cand']) if len(d['t_cand'])
                                   else np.zeros(0, np.float32))
        out[f'{fname}||obj'] = (np.concatenate(d['obj']) if len(d['obj'])
                                else np.zeros(0, np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print(f'[ORACLE] wrote {path}: {len(REC)} pieces', flush=True)
