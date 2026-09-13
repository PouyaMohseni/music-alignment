"""Fig. 2: (a) robustness to added noise, (b) synthetic validation does not rank.

Panel (a) replaces an earlier bar chart of five accuracies, every one of which
already appears in Table 2. A chart of numbers a table states more precisely is
not worth a figure; the noise ladder is a curve, is reported nowhere else, and
shows the shape the text can only assert, that the scorer's advantage shrinks
as the detector's own boxes stop being trustworthy.

(a) 80 held-out pieces rendered with a measured room impulse response and
    additive noise, 40,594 onsets per level (snr_ladder.py).
(b) Each point is one of 48 trained decision models, accuracy on the
    synthesised validation split against accuracy on the real recordings.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = '/project/def-ichiro/pmohseni/music-alignment/results'
SNRS = ('12', '6', '3', '0.5')
SERIES = [('argmax', 0, '#b8b8b8', 'o', '-'),
          ('transition', 1, '#8c8c8c', 's', '-'),
          ('no image feat.', 2, '#7fb2ae', '^', '--'),
          ('CANDOR', 3, '#1b7f79', 'D', '-')]

plt.rcParams.update({
    'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 6.2, 'axes.titlesize': 7.8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'font.family': 'sans-serif',
})


def ladder():
    f = sorted(glob.glob(f'{RES}/snrlad-*.log'), key=os.path.getmtime)[-1]
    rows = {}
    for line in open(f):
        m = re.match(r'\s*([\d.]+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
        if m and m.group(1) in SNRS:
            rows[m.group(1)] = [float(v) for v in m.groups()[1:]]
    return rows


def scatter_points():
    pts = []
    for f in sorted(glob.glob(f'{RES}/lopoall_*-2895940.log')):
        for line in open(f):
            m = re.search(r'(\S+\.pt)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+)', line)
            if not m:
                continue
            try:
                v, r = float(m.group(2)), float(m.group(4))
            except ValueError:
                continue
            if v == v and r > 60.0:
                pts.append((v, r))
    return np.array(pts)


fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.18), dpi=400)

rows = ladder()
xs = np.arange(len(SNRS))
for label, k, col, mk, ls in SERIES:
    ax[0].plot(xs, [rows[s][k] for s in SNRS], color=col, marker=mk, ls=ls,
               ms=3.0, lw=1.0, label=label, zorder=3)
ax[0].set_xticks(xs)
ax[0].set_xticklabels(SNRS)
ax[0].set_xlabel('SNR (dB)')
ax[0].set_ylabel('onsets $\\leq$0.5 s (\\%)' if False else 'onsets $\\leq$0.5 s (%)')
ax[0].set_title('(a) added noise', pad=2.5)
ax[0].grid(lw=0.4, color='#dddddd', zorder=0)
ax[0].set_axisbelow(True)
ax[0].legend(frameon=False, handlelength=1.5, labelspacing=0.18,
             borderpad=0.1, loc='lower left')
for sp in ('top', 'right'):
    ax[0].spines[sp].set_visible(False)

p = scatter_points()
ax[1].scatter(p[:, 0], p[:, 1], s=7, facecolor='#1b7f79', edgecolor='none',
              alpha=0.75, zorder=3)
r = np.corrcoef(p[:, 0], p[:, 1])[0, 1]
b, a = np.polyfit(p[:, 0], p[:, 1], 1)
xx = np.linspace(p[:, 0].min(), p[:, 0].max(), 2)
ax[1].plot(xx, a + b * xx, color='#c0392b', lw=1.0, zorder=4)
ax[1].text(0.04, 0.94, f'$r={r:.2f}$, $n={len(p)}$', transform=ax[1].transAxes,
           ha='left', va='top', fontsize=6.4)
ax[1].set_xlabel('synthesised validation (%)')
ax[1].set_ylabel('real recordings (%)')
ax[1].set_title('(b) model selection', pad=2.5)
ax[1].grid(lw=0.4, color='#dddddd', zorder=0)
ax[1].set_axisbelow(True)
for sp in ('top', 'right'):
    ax[1].spines[sp].set_visible(False)

fig.tight_layout(pad=0.2, w_pad=1.4)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'fig_analysis.{ext}'), bbox_inches='tight')
print(f'wrote fig_analysis, ladder {sorted(rows)}, {len(p)} checkpoints, r={r:.3f}')
