"""Fig. 2: (a) the selection ladder, (b) synthetic validation does not rank.

Replaces an earlier version whose panel (a) contrasted the scorer with and
without neighbourhood features. That contrast is retracted: three image
representations, including none at all, score 90.9 on the real recordings, so a
figure built on it argued the opposite of the paper.

(a) Every step from the degenerate selector to the ceiling, on the real
    recordings: confidence only, plus the parameter-free prior, plus the
    scorer, the same scorer handed the true history (teacher forcing), and the
    causal oracle over the identical hypotheses. The gap between the third and
    fourth bars is drift; between the fourth and fifth, ranking.
(b) Each point is one of 48 trained selectors: accuracy on synthesised
    validation audio against accuracy on the real recordings. Pearson r is
    computed over the points shown.
"""
from __future__ import annotations

import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = '/project/def-ichiro/pmohseni/music-alignment/results'

# real recordings, CYOLO-SB hypotheses, all cross-validated per piece
LADDER = [                     # drawn bottom-up, so ceiling last
    ('confidence only', 79.97, '#b8b8b8'),
    ('+ transition prior', 86.48, '#8c8c8c'),
    ('CANDOR', 90.92, '#1b7f79'),
    ('given true history', 98.00, '#7fb2ae'),
    ('causal oracle', 99.57, '#d9d9d9'),
]

plt.rcParams.update({
    'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.titlesize': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'font.family': 'sans-serif',
})


def scatter_points():
    """valid, room for every checkpoint that trained and decoded sanely."""
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
            if v == v and r > 60.0:      # exclude load failures and the
                pts.append((v, r))       # collapsed audio-swap runs
    return np.array(pts)


fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.28), dpi=400)

# (a) the ladder
# horizontal: five labels do not fit side by side under a 1.6 in axis
names = [n for n, _, _ in LADDER][::-1]
vals = [v for _, v, _ in LADDER][::-1]
cols = [c for _, _, c in LADDER][::-1]
ys = np.arange(len(vals))
ax[0].barh(ys, vals, color=cols, height=0.68, zorder=3, edgecolor='none')
for y, v in zip(ys, vals):
    ax[0].text(v + 0.7, y, f'{v:.1f}', ha='left', va='center', fontsize=6.4)
ax[0].set_yticks(ys)
ax[0].set_yticklabels(names, fontsize=6.4)
ax[0].set_xlim(74, 107)
ax[0].set_xticks([80, 90, 100])
ax[0].set_xlabel('onsets within 0.5 s (%)')
ax[0].set_title('(a) the selection ceiling', pad=3)
ax[0].grid(axis='x', lw=0.4, color='#dddddd', zorder=0)
ax[0].set_axisbelow(True)
for s in ('top', 'right'):
    ax[0].spines[s].set_visible(False)

# (b) validation does not rank
p = scatter_points()
ax[1].scatter(p[:, 0], p[:, 1], s=9, facecolor='#1b7f79', edgecolor='none',
              alpha=0.75, zorder=3)
r = np.corrcoef(p[:, 0], p[:, 1])[0, 1]
b, a = np.polyfit(p[:, 0], p[:, 1], 1)
xx = np.linspace(p[:, 0].min(), p[:, 0].max(), 2)
ax[1].plot(xx, a + b * xx, color='#c0392b', lw=1.0, zorder=4)
ax[1].text(0.04, 0.93, f'$r = {r:.2f}$   $n = {len(p)}$', transform=ax[1].transAxes,
           ha='left', va='top', fontsize=6.8)
ax[1].set_xlabel('synthesised validation (%)')
ax[1].set_ylabel('real recordings (%)')
ax[1].set_title('(b) synthetic vs. real', pad=3)
ax[1].grid(lw=0.4, color='#dddddd', zorder=0)
ax[1].set_axisbelow(True)
for s in ('top', 'right'):
    ax[1].spines[s].set_visible(False)

fig.tight_layout(pad=0.25, w_pad=1.6)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'fig_analysis.{ext}'), bbox_inches='tight')
print(f'wrote fig_analysis, {len(p)} checkpoints, r={r:.3f}')
