"""Fig. 1: an onset where the detector's argmax jumps staff and the decoder does not.

Crop of the padded score page with the frame's top candidates (grey), the
detector's argmax (red), our selection (teal) and the ground truth (black
ring). Candidate x in the dump is unrolled; the page position is recovered with
cyolo's own staff tables, as in extensions/analysis/dino_annotate.py. The frame
is picked automatically: argmax error > 2 s on the wrong staff, ours < 0.1 s,
the two staves close enough to share one crop.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
sys.path.insert(0, '/scratch/pmohseni/datasets/cyolo_score_following')
from extensions.analysis.dino_annotate import _data_utils
from extensions.analysis.musical_cases import FPS, load_traj

T = '/scratch/pmohseni/omr/traj'
DUMP = '/scratch/pmohseni/omr/candf256/room.npz'
DATA = '/scratch/pmohseni/datasets/cyolo_data/msmd/msmd_rp'
OUT = '/project/def-ichiro/pmohseni/music-alignment/paper/icassp2027-select/figs/fig_example'
TOPN = 40


def staff_tables(coords, page):
    pc = coords[coords[:, -1] == page]
    sc = sorted(np.unique(pc[:, 0]))
    max_xes = [0]
    for c in sc:
        max_xes.append(max(pc[pc[:, 0] == c, 1]))
    return np.asarray(sc, np.float32), np.asarray(np.cumsum(max_xes)[:-1], np.float32)


def pick(base, ours):
    best = None
    for key in sorted(set(base) & set(ours)):
        b, o = base[key], ours[key]
        fb = {int(f): i for i, f in enumerate(b['frame'])}
        for j, f in enumerate(o['frame']):
            i = fb.get(int(f))
            if i is None:
                continue
            eb = abs(b['t_pred'][i] - b['t_gt'][i]) / FPS
            eo = abs(o['t_pred'][j] - o['t_gt'][j]) / FPS
            dy = abs(b['y_pred'][i] - o['y_gt'][j])
            if eb > 2.0 and eo < 0.1 and b['staff_pred'][i] != b['staff_gt'][i] and 80 < dy < 330:
                cand = (dy, key, int(f), i, j, eb, eo)
                if best is None or cand < best:
                    best = cand
    return best


def main():
    base = load_traj(f'{T}/baseline_room.traj.npz')
    ours = load_traj(f'{T}/nbrp64s0_room.traj.npz')
    dy, key, frame, i, j, eb, eo = pick(base, ours)
    piece, page = key.rsplit('_page_', 1)
    page = int(page)
    print(f'{key} frame {frame}: argmax error {eb:.2f} s, ours {eo:.3f} s')

    z = np.load(DUMP)
    fr, lens = z[f'{key}||frame'], z[f'{key}||lens']
    k = int(np.nonzero(fr == frame)[0][0])
    off = np.concatenate([[0], np.cumsum(lens)])
    cand = z[f'{key}||cand'][off[k]:off[k + 1]]

    du = _data_utils()
    padded, _, _, coords, _, _, _, _, _, _ = du.load_piece(DATA, piece)
    sc, aps = staff_tables(coords, page)
    y = cand[:, 1]
    x = cand[:, 0] - aps[np.argmin(np.abs(sc[None, :] - y[:, None]), 1)]
    w, h = cand[:, 2], cand[:, 3]

    b, o = base[key], ours[key]
    xb, yb = b['x_pred'][i], b['y_pred'][i]
    xo, yo = o['x_pred'][j], o['y_pred'][j]
    xg, yg = o['x_gt'][j], o['y_gt'][j]
    io = int(np.argmin((x - xo) ** 2 + (y - yo) ** 2))

    img = padded[page]
    xs = [xb, xo, xg]
    ys = [yb, yo, yg]
    # MSMD rasterises a page at 1181x835 and no higher-resolution source
    # ships with it, so the only way to put enough pixels behind a 3.35 in
    # figure is to crop wide: 300 px blown up to column width is 90 dpi and
    # the noteheads turn to mush. ~700 px lands near 210 dpi.
    cx = 0.5 * (min(xs) + max(xs))
    x0, x1 = max(cx - 350, 0), min(cx + 350, img.shape[1])
    if x1 - x0 < 700:
        x0, x1 = max(x1 - 700, 0), min(x0 + 700, img.shape[1])
    y0, y1 = max(min(ys) - 40, 0), min(max(ys) + 40, img.shape[0])

    fig, ax = plt.subplots(figsize=(3.35, 3.35 * (y1 - y0) / (x1 - x0)))
    ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='antialiased')
    # outlined boxes at this scale vanished into the staff lines; a translucent
    # fill is what makes "the answer was in the set" visible at column width
    for r in range(min(TOPN, len(x)))[::-1]:
        ax.add_patch(Rectangle((x[r] - w[r] / 2, y[r] - h[r] / 2), w[r], h[r],
                               facecolor='#f2b705', edgecolor='none',
                               alpha=0.16 + 0.34 * (1 - r / TOPN)))
    ax.add_patch(Rectangle((x[0] - w[0] / 2, y[0] - h[0] / 2), w[0], h[0], fill=False,
                           lw=1.5, ec='#C0392B', label=f'confidence only, {eb:.1f}\u2009s off'))
    ax.add_patch(Rectangle((x[io] - w[io] / 2, y[io] - h[io] / 2), w[io], h[io], fill=False,
                           lw=1.5, ec='#138D90', label=f'CANDOR, {eo:.2f}\u2009s off'))
    ax.add_patch(Circle((xg, yg), 14, fill=False, lw=1.2, ec='k', label='true position'))
    ax.plot([], [], marker='s', ls='none', ms=4, color='#f2b705', alpha=0.7, label=f'{TOPN} note candidates')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.8,
              frameon=False, handlelength=1.1, columnspacing=0.9,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.0)
    fig.savefig(OUT + '.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig(OUT + '.png', dpi=400, bbox_inches='tight', pad_inches=0.01)
    print('wrote', OUT + '.pdf')


if __name__ == '__main__':
    main()
