"""Turn decoded trajectories into musical cases: what the tracker did, and why.

pct@0.5s counts frames. It cannot distinguish a half-second wobble in a dense
run from a confident commitment to the wrong repeat of a phrase, and those two
failures call for completely different fixes. This module recovers the musical
context of every error:

  repeat        the predicted bar has the SAME pitch content as the true bar, so
                the image the tracker was looking at genuinely does not identify
                the position. This is an ambiguity in the task, not a mistake in
                the model, and no better selector over one frame can fix it.
  wrong_system  landed on a different staff line altogether
  drift         right system, within a couple of bars, timing has slipped
  gross         right system, far away

Bars are assigned GEOMETRICALLY, by which bar box on that page contains the
predicted point, so nothing depends on how system indices happen to be numbered.
"""
from __future__ import annotations

import collections
import os

import numpy as np

FPS = 22050.0 / 1102.0
TH_SEC = 0.5
DATA = '/scratch/pmohseni/datasets/cyolo_data/msmd/msmd_rp'


def load_piece(name, tier='room'):
    d = np.load(os.path.join(DATA, f'{name}_{tier}.npz'), allow_pickle=True)
    return dict(sheets=d['sheets'], bars=list(d['bars']),
                systems=list(d['systems']), coords=list(d['coords']),
                wav=os.path.join(DATA, f'{name}_{tier}.wav'))


def load_traj(path):
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        page, field = k.split('||')
        out.setdefault(page, {})[field] = z[k]
    return out


def merge_pages(traj, piece_name):
    """Pages are recorded separately; a piece is their union in audio order."""
    parts = [(p, d) for p, d in traj.items() if p.startswith(piece_name + '_page_')]
    if not parts:
        return None
    keys = parts[0][1].keys()
    out = {k: np.concatenate([d[k] for _, d in parts]) for k in keys}
    out['page'] = np.concatenate(
        [np.full(len(d['frame']), int(p.rsplit('_', 1)[1])) for p, d in parts])
    o = np.argsort(out['frame'], kind='stable')
    return {k: v[o] for k, v in out.items()}


def _xyxy(b):
    return (b['x'] - b['w'] / 2.0, b['y'] - b['h'] / 2.0,
            b['x'] + b['w'] / 2.0, b['y'] + b['h'] / 2.0)


def assign_bar(page, x, y, bars):
    """Index into `bars` of the box on this page containing (x, y); else the
    nearest box on the page by centre distance; -1 if the page has none."""
    best, bestd = -1, None
    for i, b in enumerate(bars):
        if int(b['page_nr']) != int(page):
            continue
        x0, y0, x1, y1 = _xyxy(b)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return i
        d = (b['x'] - x) ** 2 + (b['y'] - y) ** 2
        if bestd is None or d < bestd:
            best, bestd = i, d
    return best


def bar_signatures(coords, bars):
    """Pitch content per bar, as a rhythm-aware signature.

    Two bars count as the same music when the same pitches arrive at the same
    offsets within the bar. Onsets are quantised to 50 ms so an expressive
    performance of the same written bar still matches itself.
    """
    per = collections.defaultdict(list)
    for c in coords:
        per[int(c['bar_idx'])].append((float(c['onset']), int(c['pitch'])))
    sig = {}
    for b, notes in per.items():
        notes.sort()
        t0 = notes[0][0]
        sig[b] = tuple(sorted((round((t - t0) * 20) / 20.0, p) for t, p in notes))
    return sig


def repeat_groups(sig, min_notes=3):
    """bar_idx -> group id, for bars whose content appears more than once."""
    by = collections.defaultdict(list)
    for b, s in sig.items():
        if len(s) >= min_notes:
            by[s].append(b)
    groups, gid = {}, 0
    for s, bs in by.items():
        if len(bs) > 1:
            for b in bs:
                groups[b] = gid
            gid += 1
    return groups, gid


def classify(tr, piece):
    """Per scored frame: error in seconds and, where it exceeds the threshold,
    which musical failure it is."""
    bars, coords = piece['bars'], piece['coords']
    sig = bar_signatures(coords, bars)
    groups, _ = repeat_groups(sig)
    # bar_idx in coords indexes the same list as `bars`
    err = np.abs(tr['t_pred'] - tr['t_gt']) / FPS
    gaps = np.diff(tr['frame'], prepend=tr['frame'][0])
    n = len(err)
    cat = np.array(['ok'] * n, dtype=object)
    bar_p = np.full(n, -1, np.int32)
    bar_g = np.full(n, -1, np.int32)
    for i in range(n):
        bar_p[i] = assign_bar(tr['page'][i], tr['x_pred'][i], tr['y_pred'][i], bars)
        bar_g[i] = assign_bar(tr['page'][i], tr['x_gt'][i], tr['y_gt'][i], bars)
        if err[i] <= TH_SEC:
            continue
        bp, bg = int(bar_p[i]), int(bar_g[i])
        if bp >= 0 and bg >= 0 and bp in groups and groups.get(bp) == groups.get(bg):
            cat[i] = 'repeat'
        elif int(tr['staff_pred'][i]) != int(tr['staff_gt'][i]):
            cat[i] = 'wrong_system'
        elif bp >= 0 and bg >= 0 and abs(bp - bg) <= 2:
            cat[i] = 'drift'
        else:
            cat[i] = 'gross'
    return dict(err=err, cat=cat, bar_pred=bar_p, bar_gt=bar_g, gap=gaps,
                t_audio=tr['frame'] / FPS, n_repeat_bars=len(groups))


def summarize(cats):
    c = collections.Counter(cats)
    n = sum(c.values())
    bad = n - c['ok']
    return n, bad, {k: (v, 100.0 * v / max(bad, 1))
                    for k, v in c.items() if k != 'ok'}
