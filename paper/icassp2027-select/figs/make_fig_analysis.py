"""Fig. 2: (a) where the remaining error lives, (b) robustness to noise.

(a) Teacher-forced decomposition of the error on the real recordings
    (extensions/analysis/error_split.py; logs results/esplit-*.log and
    results/shipoff-2862886.log): deployed accuracy, accuracy when handed the
    true history, and the causal oracle over the same candidates.
(b) Held-out pieces with a real room impulse response and additive noise
    (extensions/analysis/snr_ladder.py; results/snrlad-*.log). Falls back to
    the two scorer rows of results/shipnoise-2862888.log if the ladder log is
    not there yet.
"""
from __future__ import annotations

import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = '/project/def-ichiro/pmohseni/music-alignment/results'
ORACLE = 99.57                       # causal oracle over the top-256 candidates
SPLIT = {                            # deployed, teacher-forced
    'w/o neighbourhood': (93.42, 98.07),
    'full scorer': (94.89, 98.00),
}
SNRS = ('12', '6', '3', '0.5')


def ladder():
    logs = sorted(glob.glob(f'{RES}/snrlad-*.log'), key=os.path.getmtime)
    rows = {}
    if logs:
        for line in open(logs[-1]):
            m = re.match(r'\s*([\d.]+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if m and m.group(1) in SNRS:
                rows[m.group(1)] = tuple(float(v) for v in m.groups()[1:])
    if len(rows) == len(SNRS):
        return {'detector argmax': [rows[s][0] for s in SNRS],
                'prior only': [rows[s][1] for s in SNRS],
                'featureless scorer': [rows[s][2] for s in SNRS],
                'full scorer': [rows[s][3] for s in SNRS]}
    return {'featureless scorer': [86.79, 75.78, 55.68, 36.41],
            'full scorer': [90.69, 81.48, 64.12, 43.01]}


def main():
    plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': 0.5,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 1.45), gridspec_kw={'width_ratios': [1.15, 1]})

    names = list(SPLIT)
    for k, n in enumerate(names):
        dep, tf = SPLIT[n]
        prop, rank, gap = tf - dep, ORACLE - tf, 100 - ORACLE
        left = 0.0
        for v, c, lab in ((prop, '#C0392B', 'propagation'), (rank, '#2E86C1', 'ranking'),
                          (gap, '#AAB7B8', 'oracle gap')):
            a.barh(k, v, left=left, color=c, height=0.55, label=lab if k == 0 else None)
            left += v
        a.text(left + 0.1, k, f'{100 - dep:.2f}', va='center', fontsize=6.5)
    a.set_yticks(range(len(names)))
    a.set_yticklabels(names)
    a.invert_yaxis()
    a.set_xlim(0, 11.5)
    a.set_xticks([0, 2, 4, 6])
    a.set_xlabel('error at 0.5 s (points)')
    a.legend(fontsize=5.6, frameon=False, loc='center right', handlelength=1.0,
             borderaxespad=0.2)
    a.set_title('(a) remaining error', fontsize=7)

    rows = ladder()
    style = {'detector argmax': ('0.6', 'o'), 'prior only': ('#7F8C8D', 's'),
             'featureless scorer': ('#2E86C1', '^'), 'full scorer': ('#138D90', 'D')}
    xs = range(len(SNRS))
    for n, v in rows.items():
        c, m = style[n]
        b.plot(xs, v, color=c, marker=m, ms=2.6, lw=0.9, label=n)
    b.set_xticks(list(xs))
    b.set_xticklabels([f'{s}' for s in SNRS])
    b.set_xlabel('SNR (dB)')
    b.set_ylabel('onsets within 0.5 s (%)')
    b.legend(fontsize=5.2, frameon=False, loc='upper right', handlelength=1.2,
             borderaxespad=0.2)
    b.grid(alpha=0.25, lw=0.4)
    b.set_title('(b) held-out, added noise', fontsize=7)

    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(f'{HERE}/fig_analysis.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig(f'{HERE}/fig_analysis.png', dpi=220, bbox_inches='tight', pad_inches=0.01)
    print('wrote', f'{HERE}/fig_analysis.pdf', '| ladder rows:', list(rows))


if __name__ == '__main__':
    main()
