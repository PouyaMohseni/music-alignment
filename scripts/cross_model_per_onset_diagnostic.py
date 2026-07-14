"""Cross-model per-onset comparison: does the repeat-ambiguity correlation
found in OUR model's per-onset errors also show up in the OFFICIAL CB_TA
pretrained model's errors? If yes, it's a property of the TASK/dataset, not
an artifact of our specific architecture/training. If no, it's more
specific to us.

Uses CPJKU's own raw per-onset diffs (results/cpjku_official_pretrained/raw_onset_errors.json,
produced by eval_model.py --dump_raw_onsets, keyed by piece_page name, values
in chronological onset order) cross-referenced against OUR noteheads.npz for
the SAME piece (same onset_sec/midi_pitch/strip_x, same chronological order)
to build the identical is_repeat/density/rel_pos covariates used in our own
per-onset diagnostic.

Only uses pieces where CPJKU's pipeline produced exactly ONE page (avoids
the multi-page alignment complexity for wide pieces) AND where the onset
COUNT matches our own noteheads.npz exactly (a cheap, conservative sanity
check that the two onset sequences correspond 1:1 in the same order --
skips and reports the mismatch rate rather than silently misaligning).

    python scripts/cross_model_per_onset_diagnostic.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

import numpy as np

from mymodel.d2_midi_privileged.repeat_labels import find_repeat_groups


def main():
    raw_path = Path('results/cpjku_official_pretrained/raw_onset_errors.json')
    raw = json.load(open(raw_path))

    # group by base piece id, keep only single-page pieces
    by_base = defaultdict(list)
    for key, diffs in raw.items():
        m = re.match(r'^(.*)_page_(\d+)$', key)
        if not m:
            continue
        base, page = m.group(1), int(m.group(2))
        by_base[base].append((page, diffs))

    single_page = {base: pages[0][1] for base, pages in by_base.items() if len(pages) == 1}
    print(f'{len(raw)} piece-pages total, {len(by_base)} base pieces, '
          f'{len(single_page)} single-page (usable)')

    proc = Path('data/MSMD/processed')
    w_scale = 4  # matches v13/v14/v15's pipeline convention used throughout this session

    matched, mismatched = [], []
    all_err, all_repeat, all_density, all_relpos = [], [], [], []

    for pid, diffs in single_page.items():
        npz_path = proc / pid / 'noteheads.npz'
        if not npz_path.exists():
            continue
        notes = np.load(npz_path)
        onset_sec = notes['onset_sec']
        midi_pitch = notes['midi_pitch']
        strip_x = notes['strip_x']

        if len(diffs) != len(onset_sec):
            mismatched.append((pid, len(diffs), len(onset_sec)))
            continue
        matched.append(pid)

        diffs = np.asarray(diffs, dtype=np.float64)
        order = np.argsort(onset_sec, kind='stable')
        onset_sorted = onset_sec[order]
        pitch_sorted = midi_pitch[order]
        cols_sorted = np.round(strip_x[order] / w_scale).astype(np.int64)
        onset_frames = np.round(onset_sorted * 20).astype(np.int64)  # fps=20, matches our convention

        groups = find_repeat_groups(onset_frames, pitch_sorted.astype(np.int64), cols_sorted, k=5)
        col_alt = set()
        for c, alts in groups.items():
            col_alt.add(c)
            col_alt.update(alts)
        is_repeat = np.array([int(c) in col_alt for c in cols_sorted])

        density = np.zeros(len(onset_sorted))
        for i, t0 in enumerate(onset_sorted):
            density[i] = np.sum(np.abs(onset_sorted - t0) <= 2.0) / 4.0

        dur = max(float(onset_sorted[-1] - onset_sorted[0]), 1e-6)
        rel_pos = (onset_sorted - onset_sorted[0]) / dur

        all_err.append(diffs)
        all_repeat.append(is_repeat)
        all_density.append(density)
        all_relpos.append(rel_pos)

    print(f'matched (onset counts agree): {len(matched)} pieces')
    print(f'mismatched (skipped): {len(mismatched)} pieces')
    if mismatched:
        print('  sample mismatches:', mismatched[:5])

    if not all_err:
        print('No matched pieces -- cannot compute correlations.')
        return

    all_err = np.concatenate(all_err)
    all_repeat = np.concatenate(all_repeat)
    all_density = np.concatenate(all_density)
    all_relpos = np.concatenate(all_relpos)

    print(f'\n=== OFFICIAL CB_TA pretrained: aggregate over {len(all_err)} onsets '
          f'across {len(matched)} pieces ===')
    print(f'mean err (repeat-ambiguous onsets):     {all_err[all_repeat].mean():.3f}s  (n={all_repeat.sum()})')
    print(f'mean err (non-repeat onsets):            {all_err[~all_repeat].mean():.3f}s  (n={(~all_repeat).sum()})')

    dens_median = np.median(all_density)
    sparse = all_density < dens_median
    print(f'mean err (sparse, density<median={dens_median:.2f}): {all_err[sparse].mean():.3f}s  (n={sparse.sum()})')
    print(f'mean err (dense, density>=median):       {all_err[~sparse].mean():.3f}s  (n={(~sparse).sum()})')

    print(f'\ncorrelation(err, is_repeat):  {np.corrcoef(all_err, all_repeat.astype(float))[0,1]:.3f}')
    print(f'correlation(err, density):    {np.corrcoef(all_err, all_density)[0,1]:.3f}')
    print(f'correlation(err, rel_pos):    {np.corrcoef(all_err, all_relpos)[0,1]:.3f}')

    print('\n=== THIS IS THE KEY COMPARISON: does the same pattern hold for our own model? ===')
    print('Compare against our own model\'s per-onset correlations from')
    print('results/per_onset_diagnostic_full.json via --analyze_only.')


if __name__ == '__main__':
    main()
