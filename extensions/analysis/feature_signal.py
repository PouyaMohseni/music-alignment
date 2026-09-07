"""Before retraining: does a candidate feature actually separate right from wrong?

The index prior failed, but it failed as a REPLACEMENT for the pixel prior. The
coefficient-of-variation measurement that motivated it stands (0.425 in pixels
against 0.256 in note index), so the information may still be real and only the
delivery wrong -- as an extra per-candidate FEATURE the selector could use both.

Testing that by retraining costs an hour. Testing whether the signal exists at
all costs minutes: for every scored frame, rank the candidates by each feature
and ask how often the correct one comes first, and what its AUC is against the
incorrect ones. A feature that cannot separate them alone will not rescue a
model that already has the pixel version.

Also reports the ensemble question: how much do the existing selectors disagree?
If they agree almost everywhere, averaging them cannot help.
"""
import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.note_index import build_bins, to_index
from extensions.analysis.offline_decode import TH, load

REF = 5.0
FWD_PX = 6.0


def auc(pos, neg):
    """P(a correct candidate scores above an incorrect one), rank-based."""
    if pos.size == 0 or neg.size == 0:
        return np.nan
    a = np.concatenate([pos, neg])
    r = np.empty(a.size)
    r[np.argsort(a)] = np.arange(a.size)
    return float((r[:pos.size].sum() - pos.size * (pos.size - 1) / 2)
                 / (pos.size * neg.size))


for tier, path in (('room', '/scratch/pmohseni/omr/cand_test/room.npz'),
                   ('do', '/scratch/pmohseni/omr/cand_test/do.npz')):
    pages = load(path)
    bins = build_bins(pages, bin_px=8.0)
    acc = {'objectness': [], '|dx| pixels': [], '|dx| index': [],
           'dx pixels signed': [], 'dx index signed': []}
    first = {k: 0 for k in acc}
    n = 0
    for p in pages:
        ctr = bins[p['name']]
        xp = ip = fp = None
        for i, c in enumerate(p['cand']):
            if c.shape[0] < 2:
                continue
            err = np.abs(c[:, 5] - p['t_gt'][i])
            good = err <= TH
            if not good.any() or good.all():
                # nothing to separate on this frame
                j = int(np.argmin(err))
                xp, fp = float(c[j, 0]), int(p['frame'][i])
                ip = float(to_index(c[j:j + 1, 0], ctr)[0])
                continue
            n += 1
            df = (int(p['frame'][i]) - fp) if fp is not None else REF
            k = np.clip(max(df, 1) / REF, 0.2, 8.0)
            pos_idx = to_index(c[:, 0], ctr)
            feats = {'objectness': c[:, 4]}
            if xp is not None:
                dpx = c[:, 0] - xp
                dix = pos_idx - ip
                feats['|dx| pixels'] = -np.abs(dpx - FWD_PX * k)
                feats['|dx| index'] = -np.abs(dix - 0.5 * k)
                feats['dx pixels signed'] = dpx
                feats['dx index signed'] = dix
            for key, v in feats.items():
                acc[key].append(auc(v[good], v[~good]))
                first[key] += int(good[int(np.argmax(v))])
            j = int(np.argmin(err))
            xp, fp = float(c[j, 0]), int(p['frame'][i])
            ip = float(pos_idx[j])
    print(f'=== {tier}: {n} frames where candidates actually differ ===', flush=True)
    print('%-20s %8s %10s' % ('feature', 'AUC', 'top-1 hit'), flush=True)
    for key in ('objectness', '|dx| pixels', '|dx| index',
                'dx pixels signed', 'dx index signed'):
        v = np.array([x for x in acc[key] if np.isfinite(x)])
        if v.size:
            print('%-20s %8.3f %9.1f%%' % (key, v.mean(), 100 * first[key] / max(n, 1)),
                  flush=True)
    print('  AUC 0.5 = no signal; higher means the feature alone ranks correct '
          'candidates above incorrect ones\n', flush=True)
