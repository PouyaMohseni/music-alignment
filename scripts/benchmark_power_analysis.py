"""Statistical power of the MSMD-Rec real-audio score-following benchmark.

WHAT THIS COMPUTES, AND WHY IT IS NOT ALREADY IN THE LITERATURE
---------------------------------------------------------------
Every real-audio score-following result in the literature is a single pooled
percentage over a few thousand note onsets drawn from ~16 pieces.  Onsets inside
a piece are not independent samples: they share a performer, a take, a room, an
engraving, and a tracker state that persists across the page.  Pooling them and
reading the resulting 4-digit number as if it had onset-level precision
overstates the resolution of the benchmark by roughly two orders of magnitude.
This script quantifies by how much, and converts that into a prescription: how
many pages a benchmark needs before a k-point claim is supportable.

Ancestry, to cite rather than claim:
  * Urbano, Marrero & Martin, ISMIR 2011 -- evaluation stability / power in
    MIREX Audio Music Similarity.  Query-level, NOT clustered, and not on any
    audio-to-score alignment benchmark.
  * Cont, Schwarz, Schnell & Raphael, ISMIR 2007 -- already contrasts
    "piecewise precision rate" with "overall precision rate", i.e. notices that
    the unit of aggregation matters.  Does not analyse power.
  * Matchmaker, ISMIR 2025 -- reports both piece-wise and total alignment rate.
What is new here is the cluster-bootstrap / design-effect / minimum-detectable-
difference treatment applied to an audio-to-score alignment benchmark.

METRIC AND UNITS
----------------
pct@0.5s: percentage of evaluated note onsets whose tracked score position, when
mapped back through interpol_c2o, lands within 0.5 s of the true onset.  This is
column index 2 of the "Tracked Frame Ratios" / piecewise `[...]` lists that both
eval harnesses print, and it is the number every paper in the standings quotes.

  * unit of observation : one note onset (4415 of them on the room tier)
  * unit of publication : one pooled percentage
  * unit of RESAMPLING here : one PIECE (16 of them).  Not a page: a piece such
    as Chopin op. 9 spans 6 pages of the same performance take, so pages within
    a piece are not independent either.  Resampling pieces is the conservative,
    correct cluster.

The pooled score is the onset-count-weighted mean of the per-page scores.  This
is verified exactly against the harnesses' own pooled output (see
`verify_weighting`): reconstructed 71.15 / 79.95 / 86.46 vs logged 71.2 / 79.9 /
86.5 for cyolo / cyolo_sb / cyolo_sb_a on room -- agreement to the 0.1 rounding
the logs print at.

DATA PROVENANCE
---------------
Per-page scores are parsed, never re-run, from:
  * results/verify_cyolo_bar-143056.log  -- released CYOLO checkpoints
    (Henkel & Widmer 2021, Frontiers) on room and "do" (= direct out /
    di-left).  Format: "<page>:\n\tTracked Frame Ratios [.., .., pct@0.5, ..]".
  * results/eval_any-*.log, results/eval_msmd_rec-*.log -- this project's
    checkpoints run through eval_any_cpu.sh with --piecewise_stats.
    Format: "<page>: [.., .., pct@0.5, ..]  mean_err_s=..  median_err_s=..".
Per-page onset counts come from the ground truth itself:
  third_party/cpjku_unet/data/msmd/msmd_real_performances/score/<page>.npz,
  key `coords` -> one row per annotated note onset.  Total 4415, matching the
  harness.

STATISTICS
----------
Nonparametric cluster bootstrap: resample the 16 pieces with replacement, carry
each sampled piece's pages and onset weights, recompute the onset-weighted
pooled score (or the paired difference).  B = 20000 replicates, fixed seed.
CIs are percentile intervals.  Two-sided alpha = 0.05, power 0.80, so the
minimum detectable difference is MDD = (z_0.975 + z_0.80) * SE = 2.8016 * SE
(normal approximation to the bootstrap sampling distribution; the bootstrap
distributions here are close to symmetric, checked via skew in the JSON).

Usage:  python scripts/benchmark_power_analysis.py [--out results/analysis/power.json]
"""
import argparse
import json
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(REPO, 'third_party/cpjku_unet/data/msmd/msmd_real_performances')

Z_ALPHA = 1.959963985   # two-sided 95%
Z_POWER = 0.841621234   # 80% power
MDD_K = Z_ALPHA + Z_POWER   # 2.8016

TH_INDEX = {0.05: 0, 0.1: 1, 0.5: 2, 1.0: 3, 5.0: 4}


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def page_onset_counts():
    """Onsets per page = number of annotated note coordinates in the GT npz."""
    files = [l.strip()[2:] for l in open(os.path.join(REC, 'rp_split.yaml'))
             if l.strip().startswith('- ')]
    return {p: int(np.load(os.path.join(REC, 'score', p + '.npz'),
                           allow_pickle=True)['coords'].shape[0]) for p in files}


def piece_of(page):
    """Cluster id: strip the trailing _page_N.  Chopin op.9 -> one cluster of 6 pages."""
    return re.sub(r'_page_\d+$', '', page)


def _norm(page):
    """verify_cyolo_bar log names pages <piece>_room_page_N / <piece>_do_page_N."""
    return page.replace('_room_page_', '_page_').replace('_do_page_', '_page_')


def parse_verify_cyolo(path=None):
    """-> {(model, tier): {page: [pct@th ...]}} from results/verify_cyolo_bar-*.log"""
    path = path or os.path.join(REPO, 'results/verify_cyolo_bar-143056.log')
    txt = open(path, errors='ignore').read()
    out = {}
    for blk in re.split(r'^########## ', txt, flags=re.M)[1:]:
        hdr = blk.split('\n')[0].strip()
        model, tier = [s.strip() for s in hdr.split('/')]
        tier = {'do': 'di-left'}.get(tier, tier)
        rows = re.findall(r'^(\S+):\n\tTracked Frame Ratios \[([^\]]+)\]', blk, flags=re.M)
        if rows:
            out[(model, tier)] = {_norm(k): [float(x) for x in v.split(',')] for k, v in rows}
    return out


def parse_piecewise_log(path):
    """-> (label, tier, {page: [pct@th ...]}) from an eval_any / eval_msmd_rec log."""
    txt = open(path, errors='ignore').read()
    m = re.search(r'^experiment=(\S+)\s+(?:tier=|condition=)(\S+)', txt, flags=re.M)
    label = m.group(1) if m else os.path.basename(path)
    tier = m.group(2) if m else '?'
    rows = re.findall(r'^(\S+): \[([^\]]+)\]\s+mean_err_s=', txt, flags=re.M)
    return label, tier, {k: [float(x) for x in v.split(',')] for k, v in rows}


# --------------------------------------------------------------------------- #
# cluster bootstrap
# --------------------------------------------------------------------------- #
class Benchmark:
    """Per-page scores + onset weights + piece clustering for one tier."""

    def __init__(self, weights, th=0.5):
        self.w = weights
        self.pages = sorted(weights)
        self.th = TH_INDEX[th]
        self.clusters = {}
        for p in self.pages:
            self.clusters.setdefault(piece_of(p), []).append(p)
        self.piece_ids = sorted(self.clusters)
        # index arrays: for each piece, the page rows it owns
        self.rows = {c: np.array([self.pages.index(p) for p in self.clusters[c]])
                     for c in self.piece_ids}
        self.wv = np.array([self.w[p] for p in self.pages], dtype=float)

    def vec(self, table):
        missing = set(self.pages) - set(table)
        if missing:
            raise KeyError(f'missing pages: {sorted(missing)[:3]}')
        return np.array([table[p][self.th] for p in self.pages], dtype=float)

    def pooled(self, s, rows=None):
        if rows is None:
            return float(self.wv @ s / self.wv.sum())
        return float(self.wv[rows] @ s[rows] / self.wv[rows].sum())

    def draw(self, rng, m=None, B=20000):
        """B bootstrap replicates of row-index arrays, each from m resampled pieces."""
        m = m or len(self.piece_ids)
        pid = np.array(self.piece_ids)
        idx = rng.integers(0, len(pid), size=(B, m))
        return [np.concatenate([self.rows[pid[j]] for j in row]) for row in idx]

    def boot_stat(self, fn, rng, m=None, B=20000):
        return np.array([fn(rows) for rows in self.draw(rng, m, B)])


def ci(a, lo=2.5, hi=97.5):
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


def skew(a):
    a = np.asarray(a, float)
    s = a.std()
    return float(((a - a.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'results/analysis/benchmark_power.json'))
    ap.add_argument('--B', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20260807)
    ap.add_argument('--tier', default='room')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    W = page_onset_counts()
    bm = Benchmark(W, th=0.5)
    N_ONSETS = int(bm.wv.sum())
    R = {'meta': {'tier': args.tier, 'metric': 'pct@0.5s', 'B': args.B, 'seed': args.seed,
                  'n_pages': len(bm.pages), 'n_pieces': len(bm.piece_ids),
                  'n_onsets': N_ONSETS,
                  'cluster': 'piece (page name with trailing _page_N stripped)',
                  'mdd_constant': MDD_K}}

    # ---------------- systems ------------------------------------------------
    sysdata = {}
    for (model, tier), tab in parse_verify_cyolo().items():
        if tier == args.tier:
            sysdata[model] = tab
    extra = {
        'R3': 'results/eval_any-111639.log',
        'N3_belief_propagation': 'results/eval_any-67035775.log',
        'MERT_B2_pitch_aux': 'results/eval_any-67035750.log',
        'B1a_mert_swap': 'results/eval_any-67035735.log',
        'B3_inr_subpixel': 'results/eval_any-67035782.log',
        'B4_temporal_consistency': 'results/eval_any-67035785.log',
        'CUNet_CB_TA': 'results/eval_msmd_rec-66908892.log',
    }
    if args.tier == 'room':
        for label, rel in extra.items():
            path = os.path.join(REPO, rel)
            if os.path.exists(path):
                _, _, tab = parse_piecewise_log(path)
                if set(bm.pages) <= set(tab):
                    sysdata[label] = tab
    R['meta']['systems'] = sorted(sysdata)

    # ---------------- verify weighting reproduces the harness ---------------
    R['verify_weighting'] = {k: round(bm.pooled(bm.vec(v)), 2) for k, v in sysdata.items()}

    # ---------------- clustering structure -----------------------------------
    onsets_by_piece = {c: int(sum(W[p] for p in bm.clusters[c])) for c in bm.piece_ids}
    R['cluster_structure'] = {
        'onsets_per_piece': onsets_by_piece,
        'pages_per_piece': {c: len(v) for c, v in bm.clusters.items()},
        'largest_piece_share_pct': round(100 * max(onsets_by_piece.values()) / N_ONSETS, 1),
        'largest_piece': max(onsets_by_piece, key=onsets_by_piece.get),
        'composer_share_pct': None,
    }
    comp = {}
    for c, n in onsets_by_piece.items():
        comp[c.split('__')[0]] = comp.get(c.split('__')[0], 0) + n
    R['cluster_structure']['composer_share_pct'] = {
        k: round(100 * v / N_ONSETS, 1) for k, v in sorted(comp.items(), key=lambda kv: -kv[1])}

    # ---------------- absolute CIs, design effect, effective N ---------------
    vecs = {k: bm.vec(v) for k, v in sysdata.items()}
    boots = {}
    R['absolute'] = {}
    for k, s in vecs.items():
        b = bm.boot_stat(lambda rows, s=s: bm.pooled(s, rows), rng, B=args.B)
        boots[k] = b
        p = bm.pooled(s) / 100.0
        se_naive = 100 * np.sqrt(p * (1 - p) / N_ONSETS)
        se_cl = float(b.std(ddof=1))
        deff = (se_cl / se_naive) ** 2
        R['absolute'][k] = {
            'pct_at_0.5s': round(bm.pooled(s), 2),
            'se_cluster': round(se_cl, 2),
            'ci95': [round(x, 1) for x in ci(b)],
            'ci_width': round(ci(b)[1] - ci(b)[0], 1),
            'se_naive_onset_binomial': round(se_naive, 3),
            'design_effect': round(deff, 1),
            'effective_n_onsets': round(N_ONSETS / deff, 1),
            'boot_skew': round(skew(b), 2),
        }

    # ---------------- headline published comparisons, PAIRED -----------------
    pairs = [('cyolo', 'cyolo_sb', 'Henkel&Widmer 2021: +bar/system aux supervision'),
             ('cyolo_sb', 'cyolo_sb_a', 'Henkel&Widmer 2021: +audio augmentation'),
             ('cyolo', 'cyolo_sb_a', 'Henkel&Widmer 2021: full ablation span'),
             ('CUNet_CB_TA', 'R3', 'ours: MERT + belief propagation vs CPJKU CUNet'),
             ('R3', 'cyolo_sb', 'ours vs released SOTA-1'),
             ('cyolo_sb_a', 'R3', 'released SOTA vs ours')]
    R['paired'] = {}
    for a, b_, note in pairs:
        if a not in vecs or b_ not in vecs:
            continue
        sa, sb = vecs[a], vecs[b_]
        d = bm.boot_stat(lambda rows: bm.pooled(sb, rows) - bm.pooled(sa, rows), rng, B=args.B)
        delta = bm.pooled(sb) - bm.pooled(sa)
        se = float(d.std(ddof=1))
        lo, hi = ci(d)
        # per-piece deltas (onset weighted inside piece)
        pd_ = {c: (float(bm.wv[bm.rows[c]] @ (sb - sa)[bm.rows[c]] / bm.wv[bm.rows[c]].sum()))
               for c in bm.piece_ids}
        # unpaired counterfactual: as if the two numbers came from independent samples
        se_unp = float(np.sqrt(boots[a].var(ddof=1) + boots[b_].var(ddof=1)))
        R['paired'][f'{a}__vs__{b_}'] = {
            'note': note,
            'delta_pct_points': round(delta, 2),
            'se_paired': round(se, 2),
            'ci95_paired': [round(lo, 2), round(hi, 2)],
            'significant_paired_a05': bool(lo > 0 or hi < 0),
            'boot_p_two_sided': round(2 * min((d <= 0).mean(), (d >= 0).mean()), 4),
            'mdd80_paired': round(MDD_K * se, 2),
            'se_if_unpaired': round(se_unp, 2),
            'ci95_if_unpaired': [round(delta - Z_ALPHA * se_unp, 2),
                                 round(delta + Z_ALPHA * se_unp, 2)],
            'significant_unpaired_a05': bool(abs(delta) > Z_ALPHA * se_unp),
            'mdd80_unpaired': round(MDD_K * se_unp, 2),
            'per_piece_delta': {c: round(v, 1) for c, v in sorted(pd_.items(), key=lambda kv: kv[1])},
            'n_pieces_favouring_b': int(sum(v > 0 for v in pd_.values())),
            'per_piece_delta_sd': round(float(np.std(list(pd_.values()), ddof=1)), 2),
            'piece_score_correlation': round(float(np.corrcoef(sa, sb)[0, 1]), 3),
        }

    # ---------------- aggregation convention: pooled vs piece-wise -----------
    # Cont et al. ISMIR 2007 already flags that "overall precision rate" and
    # "piecewise precision rate" are different metrics.  Here we quantify how
    # much the CHOICE alone moves a published headline delta.
    R['aggregation_convention'] = {}
    for k, s in vecs.items():
        pw = float(np.mean([bm.wv[bm.rows[c]] @ s[bm.rows[c]] / bm.wv[bm.rows[c]].sum()
                            for c in bm.piece_ids]))
        R['aggregation_convention'][k] = {
            'onset_pooled': round(bm.pooled(s), 2),
            'page_mean': round(float(s.mean()), 2),
            'piece_mean': round(pw, 2),
            'pooled_minus_piece_mean': round(bm.pooled(s) - pw, 2)}
    for a, b_, _n in pairs[:3]:
        if a in vecs and b_ in vecs:
            R['aggregation_convention'][f'DELTA {a}->{b_}'] = {
                'onset_pooled': round(R['aggregation_convention'][b_]['onset_pooled']
                                      - R['aggregation_convention'][a]['onset_pooled'], 2),
                'piece_mean': round(R['aggregation_convention'][b_]['piece_mean']
                                    - R['aggregation_convention'][a]['piece_mean'], 2)}

    # ---------------- MDD and power curve ------------------------------------
    ref = 'cyolo_sb' if 'cyolo_sb' in vecs else sorted(vecs)[0]
    se_abs = float(boots[ref].std(ddof=1))
    grid = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    pages_per_piece = len(bm.pages) / len(bm.piece_ids)
    onsets_per_piece = N_ONSETS / len(bm.piece_ids)

    # representative paired SE: the median across the published-ablation pairs we can test.
    # NOTE the paired SE is pair-dependent -- it falls as the two systems' per-piece
    # scores correlate -- so the range across pairs is reported alongside.
    # only the three CYOLO-family ablation pairs -- these are the actual published
    # ablations, and they are the right reference for "can a paper's own ablation
    # be resolved".  Cross-family pairs (ours vs theirs) are far less correlated
    # per piece and would inflate the reference SE.
    pk = [k for k in ('cyolo__vs__cyolo_sb', 'cyolo_sb__vs__cyolo_sb_a',
                      'cyolo__vs__cyolo_sb_a') if k in R['paired']]
    se_list = [R['paired'][k]['se_paired'] for k in pk]
    se_pair16 = float(np.median(se_list)) if se_list else se_abs

    curve = []
    for m in grid:
        b_abs = bm.boot_stat(lambda rows: bm.pooled(vecs[ref], rows), rng, m=m, B=4000)
        se_a = float(b_abs.std(ddof=1))
        se_p = se_pair16 * np.sqrt(len(bm.piece_ids) / m)     # 1/sqrt(m) scaling
        curve.append({'n_pieces': m,
                      'n_pages': round(m * pages_per_piece),
                      'n_onsets': round(m * onsets_per_piece),
                      'se_absolute': round(se_a, 2),
                      'mdd80_absolute_single_system': round(MDD_K * se_a, 1),
                      'mdd80_unpaired_two_systems': round(MDD_K * np.sqrt(2) * se_a, 1),
                      'se_paired': round(se_p, 2),
                      'mdd80_paired': round(MDD_K * se_p, 2)})
    R['power_curve'] = curve

    def need(delta, mode):
        """pieces required so that MDD80 <= delta, from the 1/sqrt(m) law."""
        if mode == 'paired':
            se1 = se_pair16 * np.sqrt(len(bm.piece_ids))
        elif mode == 'unpaired':
            se1 = np.sqrt(2) * se_abs * np.sqrt(len(bm.piece_ids))
        else:
            se1 = se_abs * np.sqrt(len(bm.piece_ids))
        m = (MDD_K * se1 / delta) ** 2
        return {'n_pieces': int(np.ceil(m)),
                'n_pages': int(np.ceil(m * pages_per_piece)),
                'n_onsets': int(np.ceil(m * onsets_per_piece))}

    R['requirements'] = {f'{d}pt': {mode: need(d, mode) for mode in ('paired', 'unpaired', 'absolute')}
                         for d in (2, 5, 10)}
    R['current_mdd80'] = {
        'paired': round(MDD_K * se_pair16, 2),
        'paired_range_over_testable_pairs': [round(MDD_K * min(se_list), 2),
                                             round(MDD_K * max(se_list), 2)] if se_list else None,
        'unpaired': round(MDD_K * np.sqrt(2) * se_abs, 1),
        'absolute_single_system': round(MDD_K * se_abs, 1),
        'se_paired_reference': round(se_pair16, 2),
        'se_absolute_reference': round(se_abs, 2),
        'reference_system': ref,
    }

    # required pages as a joint function of target delta and the pair's SE, because
    # MDD_paired is not a property of the benchmark alone -- it depends on how
    # correlated the two systems are across pieces.
    R['requirements_by_pair_se'] = {}
    for se16 in sorted(set(round(x, 2) for x in (se_list or [se_abs]))):
        se1 = se16 * np.sqrt(len(bm.piece_ids))
        R['requirements_by_pair_se'][f'se_paired_at_n16={se16}'] = {
            f'{d}pt': {'n_pieces': int(np.ceil((MDD_K * se1 / d) ** 2)),
                       'n_pages': int(np.ceil((MDD_K * se1 / d) ** 2 * pages_per_piece))}
            for d in (2, 5, 10)}

    # ---------------- benchmark noise vs TRAINING-SEED noise -----------------
    # Published ablations report one training seed per arm.  Sampling noise over
    # pieces is therefore only half the story.  Seed sd = 3.5 pct points was
    # measured in this project by training 9 recipes twice (MEMORY / real-audio
    # sweep); combined recipe-vs-recipe sd = 4.3.  Those were measured on models
    # scoring 15-45 on room, where a proportion's variance is near its maximum;
    # we therefore also report a variance-stabilised rescaling to the CYOLO
    # operating point via sqrt(p(1-p)).  This section is an EXTRAPOLATION using
    # our own seed measurements, not a measurement of CYOLO's seed variance,
    # which would require retraining their models and is not claimed.
    SEED_SD_MEASURED, SEED_SD_AT_P = 3.5, 0.30
    R['seed_inclusive'] = {'seed_sd_measured': SEED_SD_MEASURED,
                           'seed_sd_measured_at_score': SEED_SD_AT_P * 100,
                           'provenance': '9 recipes trained twice, this project',
                           'pairs': {}}
    for k, v in R['paired'].items():
        p = np.mean([R['absolute'][s]['pct_at_0.5s'] for s in k.split('__vs__')]) / 100
        scale = np.sqrt(p * (1 - p)) / np.sqrt(SEED_SD_AT_P * (1 - SEED_SD_AT_P))
        for tag, sd in (('raw', SEED_SD_MEASURED), ('variance_stabilised', SEED_SD_MEASURED * scale)):
            se_tot = float(np.sqrt(v['se_paired'] ** 2 + 2 * sd ** 2))
            d = v['delta_pct_points']
            R['seed_inclusive']['pairs'].setdefault(k, {})[tag] = {
                'seed_sd_per_arm': round(sd, 2),
                'se_total': round(se_tot, 2),
                'ci95': [round(d - Z_ALPHA * se_tot, 2), round(d + Z_ALPHA * se_tot, 2)],
                'significant': bool(abs(d) > Z_ALPHA * se_tot),
                'mdd80': round(MDD_K * se_tot, 2)}

    # ---------------- rank stability of the published leaderboard ------------
    board = [k for k in ('cyolo_sb_a', 'cyolo_sb', 'cyolo', 'R3', 'CUNet_CB_TA') if k in vecs]
    if len(board) >= 2:
        draws = bm.draw(rng, B=args.B)
        M = np.array([[bm.pooled(vecs[k], rows) for k in board] for rows in draws])
        order = np.argsort(-np.array([bm.pooled(vecs[k]) for k in board]))
        board = [board[i] for i in order]
        M = M[:, order]
        R['rank_stability'] = {
            'order': board,
            'p_full_order_preserved': round(float(np.mean(np.all(np.diff(M, axis=1) < 0, axis=1))), 4),
            'adjacent': {f'{board[i]}>{board[i+1]}': round(float(np.mean(M[:, i] > M[:, i + 1])), 4)
                         for i in range(len(board) - 1)},
        }

    # ---------------- leave-one-piece-out sensitivity ------------------------
    R['leave_one_piece_out'] = {}
    for a, b_, _n in pairs:
        if a not in vecs or b_ not in vecs:
            continue
        sa, sb = vecs[a], vecs[b_]
        full = bm.pooled(sb) - bm.pooled(sa)
        swing = {}
        for c in bm.piece_ids:
            keep = np.array([i for i in range(len(bm.pages)) if i not in set(bm.rows[c])])
            swing[c] = round(bm.pooled(sb, keep) - bm.pooled(sa, keep) - full, 2)
        worst = max(swing, key=lambda k: abs(swing[k]))
        R['leave_one_piece_out'][f'{a}__vs__{b_}'] = {
            'delta_full': round(full, 2), 'max_abs_swing_piece': worst,
            'max_abs_swing': swing[worst], 'all': swing}

    # ---------------- what we can and cannot say about CODA ------------------
    coda_delta = 88.3 - 79.9
    if 'cyolo_sb' in vecs and 'cyolo_sb_a' in vecs:
        se_ref = R['paired']['cyolo_sb__vs__cyolo_sb_a']['se_paired']
    else:
        se_ref = se_pair16
    R['coda'] = {
        'claim': 'CODA (ISMIR 2026) 88.3 vs cyolo_sb 79.9 on room, +8.4 points',
        'testable': False,
        'reason': ('CODA per-piece scores are not released and we have no CODA checkpoint, '
                   'so no paired test is possible.  Only the two pooled numbers exist.'),
        'unpaired_bound': {
            'delta': coda_delta,
            'se_unpaired': round(float(np.sqrt(2)) * se_abs, 2),
            'ci95': [round(coda_delta - Z_ALPHA * np.sqrt(2) * se_abs, 1),
                     round(coda_delta + Z_ALPHA * np.sqrt(2) * se_abs, 1)],
            'significant': bool(coda_delta > Z_ALPHA * np.sqrt(2) * se_abs),
            'interpretation': ('Treating the two published numbers as independent draws '
                               'over pieces -- the only inference available without CODA '
                               'per-piece scores -- the gap is NOT resolvable.')},
        'paired_counterfactual': {
            'assumed_se_paired': se_ref,
            'note': ('If CODA per-piece scores behaved like the cyolo_sb -> cyolo_sb_a pair '
                     '(the closest available analogue: same benchmark, same family, similar '
                     'gap), a +8.4 paired difference WOULD clear alpha=0.05.  We state this '
                     'as a counterfactual, not a result: it cannot be checked.'),
            'would_be_significant': bool(coda_delta > Z_ALPHA * se_ref)},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(R, open(args.out, 'w'), indent=2)
    print(f'wrote {args.out}')

    # ---------------- console summary ---------------------------------------
    print('\n=== weighting check (reconstructed pooled pct@0.5s) ===')
    for k, v in sorted(R['verify_weighting'].items(), key=lambda kv: -kv[1]):
        print(f'  {k:26s} {v}')
    print('\n=== absolute scores, piece-level cluster bootstrap ===')
    print(f'  {"system":26s} {"pct":>6s} {"se":>6s}  {"95% CI":>16s} {"deff":>7s} {"n_eff":>7s}')
    for k, v in sorted(R['absolute'].items(), key=lambda kv: -kv[1]['pct_at_0.5s']):
        print(f'  {k:26s} {v["pct_at_0.5s"]:6.1f} {v["se_cluster"]:6.2f}  '
              f'[{v["ci95"][0]:6.1f},{v["ci95"][1]:6.1f}] {v["design_effect"]:7.1f} '
              f'{v["effective_n_onsets"]:7.1f}')
    print('\n=== headline comparisons ===')
    for k, v in R['paired'].items():
        print(f'  {k:34s} d={v["delta_pct_points"]:+6.2f}  paired CI '
              f'[{v["ci95_paired"][0]:+6.2f},{v["ci95_paired"][1]:+6.2f}] '
              f'{"SIG" if v["significant_paired_a05"] else "n.s.":>4s} | '
              f'unpaired CI [{v["ci95_if_unpaired"][0]:+6.1f},{v["ci95_if_unpaired"][1]:+6.1f}] '
              f'{"SIG" if v["significant_unpaired_a05"] else "n.s."}')
    print('\n=== power curve (MDD at 80% power, alpha=0.05) ===')
    print(f'  {"pieces":>7s} {"pages":>6s} {"onsets":>7s} {"MDD abs":>8s} {"MDD unpaired":>13s} {"MDD paired":>11s}')
    for c in R['power_curve']:
        print(f'  {c["n_pieces"]:7d} {c["n_pages"]:6d} {c["n_onsets"]:7d} '
              f'{c["mdd80_absolute_single_system"]:8.1f} {c["mdd80_unpaired_two_systems"]:13.1f} '
              f'{c["mdd80_paired"]:11.2f}')
    print('\n=== pages required ===')
    for d, m in R['requirements'].items():
        print(f'  detect {d}: paired {m["paired"]["n_pages"]} pages / '
              f'{m["paired"]["n_pieces"]} pieces | unpaired {m["unpaired"]["n_pages"]} pages / '
              f'{m["unpaired"]["n_pieces"]} pieces')
    print('\n=== published ablations once single-seed training noise is included ===')
    for k, v in R['seed_inclusive']['pairs'].items():
        w = v['variance_stabilised']
        print(f'  {k:34s} se_tot={w["se_total"]:5.2f} CI [{w["ci95"][0]:+7.2f},{w["ci95"][1]:+7.2f}] '
              f'{"SIG" if w["significant"] else "n.s."}')
    print('\n=== aggregation convention (Cont 2007 pooled vs piecewise) ===')
    for k, v in R['aggregation_convention'].items():
        if k.startswith('DELTA'):
            print(f'  {k:34s} pooled={v["onset_pooled"]:+6.2f}  piece-mean={v["piece_mean"]:+6.2f}')
    if 'rank_stability' in R:
        print('\n=== rank stability ===')
        print('  order:', ' > '.join(R['rank_stability']['order']))
        print('  P(full order preserved):', R['rank_stability']['p_full_order_preserved'])
        for k, v in R['rank_stability']['adjacent'].items():
            print(f'    P({k}) = {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
