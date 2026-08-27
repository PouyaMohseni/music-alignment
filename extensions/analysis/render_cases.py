"""Figures for the musical demo: what the tracker did, on which music, and why.

Three views per case, because each answers a different question:

  path      audio time against score time. A perfect tracker is the diagonal;
            tempo shows up as its slope, and a wrong commitment shows up as a
            plateau or a jump, which an error curve alone would hide.
  error     the same thing as |error| in seconds against the 0.5 s threshold, so
            the reported metric is visible frame by frame.
  page      the score itself with both paths drawn on it, coloured by whether
            the frame was inside the threshold. This is the view that shows
            WHERE on the page the tracker went, which is the thing a number
            can never show.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from extensions.analysis.musical_cases import (FPS, TH_SEC, assign_bar, classify,
                                               load_piece, load_traj, merge_pages)

CAT_COLOR = {'repeat': '#c0392b', 'wrong_system': '#8e44ad',
             'drift': '#e67e22', 'gross': '#2c3e50', 'ok': '#27ae60'}


def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=.18, linewidth=.6)


def path_figure(piece_name, arms, piece, out, title=None):
    """arms: {label: merged trajectory}. The first is treated as the reference
    for ground truth (identical across arms by construction)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.1, 1]))
    ref = next(iter(arms.values()))
    ta = ref['frame'] / FPS
    axes[0].plot(ta, ref['t_gt'] / FPS, color='#111', lw=2.4, label='ground truth',
                 zorder=3)
    colors = ['#2980b9', '#e74c3c', '#16a085']
    for (lab, tr), c in zip(arms.items(), colors):
        axes[0].plot(tr['frame'] / FPS, tr['t_pred'] / FPS, lw=1.3, color=c,
                     alpha=.9, label=lab)
        e = np.abs(tr['t_pred'] - tr['t_gt']) / FPS
        axes[1].plot(tr['frame'] / FPS, e, lw=1.2, color=c, label=lab)
    axes[1].axhline(TH_SEC, color='#111', ls='--', lw=1, label='0.5 s threshold')
    axes[1].set_ylim(0, max(3.0, float(np.percentile(
        np.concatenate([np.abs(t['t_pred'] - t['t_gt']) / FPS
                        for t in arms.values()]), 99.5)) * 1.1))
    axes[0].set_ylabel('position in the score  (s)')
    axes[1].set_ylabel('|error|  (s)')
    axes[1].set_xlabel('time in the performance  (s)')
    axes[0].legend(frameon=False, ncol=4, fontsize=9, loc='upper left')
    axes[1].legend(frameon=False, ncol=3, fontsize=8, loc='upper left')
    for a in axes:
        _style(a)
    fig.suptitle(title or piece_name, fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def page_figure(piece_name, tr, piece, page, out, label='ours'):
    """The score page with both paths drawn where they actually went."""
    m = tr['page'] == page
    if not m.any():
        return None
    sheet = piece['sheets'][page]
    err = np.abs(tr['t_pred'] - tr['t_gt'])[m] / FPS
    fig, ax = plt.subplots(figsize=(9, 9 * sheet.shape[0] / sheet.shape[1]))
    ax.imshow(sheet, cmap='gray', vmin=0, vmax=255)
    for b in piece['bars']:
        if int(b['page_nr']) == page:
            ax.add_patch(plt.Rectangle((b['x'] - b['w'] / 2, b['y'] - b['h'] / 2),
                                       b['w'], b['h'], fill=False,
                                       edgecolor='#3498db', lw=.5, alpha=.35))
    ax.plot(tr['x_gt'][m], tr['y_gt'][m], '.', ms=3.5, color='#111',
            label='ground truth', zorder=3)
    good = err <= TH_SEC
    ax.plot(tr['x_pred'][m][good], tr['y_pred'][m][good], '.', ms=3.5,
            color='#27ae60', alpha=.85, label=f'{label}: within 0.5 s')
    ax.plot(tr['x_pred'][m][~good], tr['y_pred'][m][~good], 'x', ms=5,
            color='#c0392b', mew=1.2, label=f'{label}: outside 0.5 s')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title(f'{piece_name}  ·  page {page}  ·  '
                 f'{100 * good.mean():.1f}% within 0.5 s', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def tier_figure(piece_name, per_tier, out):
    """The SAME performance under three recording conditions."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ref = next(iter(per_tier.values()))
    ax.plot(ref['frame'] / FPS, ref['t_gt'] / FPS, color='#111', lw=2.4,
            label='ground truth', zorder=3)
    for (tier, tr), c in zip(per_tier.items(), ['#2980b9', '#e67e22', '#16a085']):
        e = np.abs(tr['t_pred'] - tr['t_gt']) / FPS
        ax.plot(tr['frame'] / FPS, tr['t_pred'] / FPS, lw=1.3, color=c, alpha=.9,
                label=f'{tier}  ({100 * (e <= TH_SEC).mean():.1f}% within 0.5 s)')
    ax.set_xlabel('time in the performance  (s)')
    ax.set_ylabel('position in the score  (s)')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    _style(ax)
    ax.set_title(f'{piece_name}: one performance, three recording conditions',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def taxonomy_figure(counts, out):
    """Where the failures live, aggregated over the test set."""
    order = ['repeat', 'wrong_system', 'drift', 'gross']
    vals = [counts.get(k, 0) for k in order]
    tot = max(sum(vals), 1)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.barh(range(len(order))[::-1], vals,
                   color=[CAT_COLOR[k] for k in order], height=.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + tot * .012, b.get_y() + b.get_height() / 2,
                f'{v}  ({100 * v / tot:.0f}%)', va='center', fontsize=9)
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels(['same music elsewhere\n(repeat ambiguity)',
                        'wrong staff line', 'timing drift\n(within 2 bars)',
                        'gross, same staff'], fontsize=9)
    ax.set_xlabel('frames outside the 0.5 s threshold')
    ax.set_xlim(0, tot * 1.18)
    _style(ax)
    ax.grid(axis='y', alpha=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
