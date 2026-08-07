"""Is a score follower's confidence calibrated against its own tracking error?

THE QUESTION
------------
A score follower that is lost but confident is worse than useless -- a page
turner will turn, an accompanist will play.  Every deployed system therefore
needs a "am I lost?" signal.  The literature has exactly two, and both are
hand-built binary rules with no calibration evaluation:

  * Brazier & Widmer, EUSIPCO 2021 -- "reliability factor" rf in {0, 1}:
    rf = 1 iff the last 30 tracked score positions fit a straight line whose
    slope lies in [0.5, 1.5] (i.e. the tracker is advancing at roughly
    real-time pace).  Binary, hand-tuned thresholds, never scored as a
    detector.
  * CODA, ISMIR 2026 -- a silence-driven "break mode": suspend tracking when
    the input is silent.  Also a hand-built rule.

Neither paper reports whether the rule actually discriminates lost frames from
tracked frames.  This script does: it scores continuous confidence signals that
the models already emit, against those two published heuristics, as detectors of
"this frame's tracking error exceeds 0.5 s" -- the same 0.5 s that defines the
benchmark's headline metric.  The contribution is therefore "we beat two
published heuristics at a task nobody has measured", not "we propose
calibration".

DEFINITIONS (exact)
-------------------
  positive class : err_seconds > 0.5, where err_seconds = err_frames / fps and
                   err_frames is |interpol_c2o(x_pred) - interpol_c2o(x_gt)| --
                   bit-for-bit the quantity both harnesses turn into pct@0.5s.
                   fps = 22050/1102 = 20.009 (CYOLO) or 20.0 (CUNet).
  evaluated frames : onset frames only, matching the published metric
                   (eval.py --only_onsets / eval_model.py --eval_onsets).
                   All-frame numbers are reported alongside as a secondary.
  AUROC          : probability that a randomly chosen lost frame is ranked as
                   less confident than a randomly chosen tracked frame.
                   Computed by rank statistic (Mann-Whitney U / (n_pos*n_neg)),
                   ties at 0.5.  Signals whose natural polarity is "higher =
                   more likely lost" (entropy) are used as-is; signals whose
                   polarity is "higher = more confident" are negated, so every
                   reported AUROC is for a LOSTNESS score.  AUROC 0.5 = useless.
  AP             : average precision (area under precision-recall), which is the
                   honest headline when lost frames are the minority class.
  95% CI         : piece-level cluster bootstrap, B replicates over the 16
                   pieces -- the same clustering as
                   scripts/benchmark_power_analysis.py, for the same reason:
                   frames inside a piece are not independent.

BASELINES IMPLEMENTED
---------------------
  brazier_rf     Brazier & Widmer's rule, ported to this data.  Least-squares
                 slope of the last W = 30 evaluated tracked positions, expressed
                 in ONSET-FRAMES OF SCORE TIME PER AUDIO FRAME, so that a
                 tracker moving through the score at real-time pace has slope
                 1.0 exactly and their [0.5, 1.5] window is the paper's window
                 without retuning.  Positions are mapped through interpol_c2o,
                 the same map the metric uses, because raw x pixels advance at a
                 rate that depends on note density and would make the [0.5, 1.5]
                 window meaningless.  Emits rf = 0 (unreliable) / 1 (reliable),
                 so it is a single operating point, not a ranking: it is scored
                 by balanced accuracy, TPR, FPR, and the AUROC its 2-point ROC
                 attains, which is what a binary rule can achieve.
                 DEVIATION, stated for honesty: their rule ran at full frame
                 rate on their own tracker; we apply it to the frames this
                 harness evaluates.  A --all_frames dump is used where available
                 so the window really is 30 consecutive audio frames.
  silence        CODA-style break-mode trigger: frame-wise RMS of the
                 performance audio, in dB relative to the piece's 95th
                 percentile, thresholded.  Reported both as a continuous
                 lostness score (so it gets a full AUROC, which is generous to
                 it) and at the fixed -40 dB operating point.

CANDIDATE MODEL SIGNALS
-----------------------
  CYOLO family  : conf1 (max objectness), 1-(conf1-conf2) top-2 margin,
                  ent_x (entropy of the objectness x-marginal).
  CUNet family  : conf_max (heatmap peak), conf_mass (peak of the normalised
                  x-marginal), margin, ent_x.
  combo         : logistic regression on the model signals, fit and scored with
                  LEAVE-ONE-PIECE-OUT cross-validation so no piece is ever in
                  its own training fold.  This is the only trained signal here;
                  everything else is parameter-free.

Usage:
    python scripts/calibration_analysis.py \
        --dump results/calibration/cyolo_sb_room_allframes.npz \
        --audio-root /scratch/pmohseni/datasets/cyolo_data/msmd/msmd_rp \
        --audio-suffix _room.wav \
        --out results/analysis/calibration_cyolo_sb_room.json
"""
import argparse
import json
import os
import re
import sys
import wave

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
def load_dump(path):
    z = np.load(path, allow_pickle=True)
    fps = float(z['__fps__'][0]) if '__fps__' in z else 20.0
    pages = {}
    for k in z.files:
        if '||' not in k:
            continue
        page, field = k.split('||')
        pages.setdefault(page, {})[field] = z[k]
    return pages, fps


def piece_of(page):
    """Cluster id.  Strips the page index and, for CYOLO page names, the tier tag
    that its npz basenames carry (<piece>_room_page_0 -> <piece>)."""
    return re.sub(r'_(room|do|di-left|rp_synth)$', '', re.sub(r'_page_\d+$', '', page))


def find_wav(root, page, suffix):
    """Both dumps name pages differently and their audio lives in different trees.
    CYOLO : page = <piece>_room_page_0, audio = <root>/<piece>_room.wav
    CUNet : page = <piece>_page_0,      audio = <root>/<piece>_page_0_room.wav
    """
    stem = re.sub(r'_page_\d+$', '', page)
    for cand in (page + suffix, stem + suffix, stem + '.wav', page + '.wav'):
        p = os.path.join(root, cand)
        if os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auroc(score, y):
    """P(score[lost] > score[tracked]); ties 0.5.  score = LOSTNESS."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(np.asarray(score, float))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rankdata(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def average_precision(score, y):
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return float('nan')
    o = np.argsort(-np.asarray(score, float), kind='mergesort')
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def binary_stats(pred_lost, y):
    y = np.asarray(y, bool)
    p = np.asarray(pred_lost, bool)
    tp, fn = int((p & y).sum()), int((~p & y).sum())
    fp, tn = int((p & ~y).sum()), int((~p & ~y).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {'tpr_recall_of_lost': round(tpr, 4), 'fpr': round(fpr, 4),
            'precision': round(tp / max(tp + fp, 1), 4),
            'balanced_accuracy': round(0.5 * (tpr + (1 - fpr)), 4),
            'auroc_of_this_operating_point': round(0.5 * (1 + tpr - fpr), 4),
            'flag_rate': round(float(p.mean()), 4)}


def ece(score01, y, bins=10):
    """Expected calibration error of a signal already mapped to [0,1] = P(lost)."""
    y = np.asarray(y, bool)
    s = np.clip(np.asarray(score01, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    tot, out = len(s), 0.0
    diag = []
    for i in range(bins):
        m = (s >= edges[i]) & (s < edges[i + 1] if i < bins - 1 else s <= 1.0)
        if not m.any():
            diag.append({'bin': [round(edges[i], 2), round(edges[i + 1], 2)], 'n': 0,
                         'mean_pred': None, 'frac_lost': None})
            continue
        mp, fl = float(s[m].mean()), float(y[m].mean())
        out += m.sum() / tot * abs(mp - fl)
        diag.append({'bin': [round(edges[i], 2), round(edges[i + 1], 2)], 'n': int(m.sum()),
                     'mean_pred': round(mp, 4), 'frac_lost': round(fl, 4)})
    return float(out), diag


# --------------------------------------------------------------------------- #
# published baselines
# --------------------------------------------------------------------------- #
def brazier_reliability(c2o_frames, window=30, lo=0.5, hi=1.5):
    """Brazier & Widmer EUSIPCO 2021 reliability factor.

    c2o_frames: the tracked position already expressed in ONSET-FRAME units
    (i.e. interpol_c2o(x_pred)), so slope = d(score time)/d(audio time) and a
    correct real-time tracker sits at 1.0.  Returns rf in {0,1}; rf = 0 means
    "unreliable" -> predicted LOST.  Frames before the window fills inherit the
    first computable value (their tracker likewise cannot judge frame 0).
    """
    n = len(c2o_frames)
    rf = np.ones(n, dtype=bool)
    t = np.arange(window, dtype=float)
    t -= t.mean()
    denom = float((t * t).sum())
    for i in range(n):
        if i + 1 < window:
            continue
        seg = np.asarray(c2o_frames[i + 1 - window:i + 1], float)
        slope = float((t * (seg - seg.mean())).sum() / denom)
        rf[i] = (lo <= slope <= hi)
    if n >= window:
        rf[:window - 1] = rf[window - 1]
    return rf


def frame_rms_db(wav_path, n_frames, hop, frame_size, sr):
    """Frame-wise RMS in dB relative to the piece's 95th percentile RMS."""
    w = wave.open(wav_path, 'rb')
    raw = w.readframes(w.getnframes())
    sw, ch = w.getsampwidth(), w.getnchannels()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    sig = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if ch > 1:
        sig = sig.reshape(-1, ch).mean(1)
    sig /= (2 ** (8 * sw - 1))
    rate = w.getframerate()
    if rate != sr:                                  # nearest-neighbour resample
        idx = (np.arange(int(len(sig) * sr / rate)) * rate / sr).astype(np.int64)
        sig = sig[np.clip(idx, 0, len(sig) - 1)]
    out = np.zeros(n_frames)
    for i in range(n_frames):
        a = i * hop
        seg = sig[a:a + frame_size]
        out[i] = np.sqrt((seg ** 2).mean()) if len(seg) else 0.0
    ref = np.percentile(out, 95) or 1e-9
    return 20 * np.log10(np.maximum(out, 1e-9) / ref)


# --------------------------------------------------------------------------- #
def logistic_loo(X, y, groups, l2=1.0, iters=300, lr=0.5):
    """Leave-one-piece-out logistic regression; returns out-of-fold P(lost)."""
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    y = np.asarray(y, float)
    oof = np.zeros(len(y))
    for g in sorted(set(groups)):
        te = np.array([gg == g for gg in groups])
        tr = ~te
        if y[tr].sum() in (0, tr.sum()):
            oof[te] = y[tr].mean()
            continue
        w = np.zeros(Xs.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Xs[tr] @ w))
            grad = Xs[tr].T @ (p - y[tr]) / tr.sum() + l2 * np.r_[w[:-1], 0] / tr.sum()
            w -= lr * grad
        oof[te] = 1 / (1 + np.exp(-Xs[te] @ w))
    return oof


def cluster_ci(fn, score, y, pieces, B=2000, seed=7):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(pieces))
    idx = {p: np.flatnonzero(np.asarray(pieces) == p) for p in uniq}
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[uniq[j]] for j in rng.integers(0, len(uniq), len(uniq))])
        v = fn(np.asarray(score)[take], np.asarray(y)[take])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [float('nan')] * 2
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--audio-root', default='/scratch/pmohseni/datasets/cyolo_data/msmd/msmd_rp')
    ap.add_argument('--audio-suffix', default='_room.wav')
    ap.add_argument('--threshold-s', type=float, default=0.5)
    ap.add_argument('--B', type=int, default=2000)
    args = ap.parse_args()

    pages, fps = load_dump(args.dump)
    R = {'meta': {'dump': args.dump, 'fps': fps, 'threshold_s': args.threshold_s,
                  'n_pages': len(pages), 'positive_class': f'err > {args.threshold_s}s'}}

    # ---- assemble a flat table, keeping per-page temporal order -------------
    cols, pieces, page_of_row, onset_mask = {}, [], [], []
    c2o_check, missing_audio = [], []
    for page in sorted(pages):
        d = pages[page]
        n = len(d['err_frames'])

        # Brazier & Widmer rf needs the tracked position in score-time units.
        # err_frames is |c2o(pred) - c2o(gt)|, so c2o(pred) itself is not stored;
        # reconstruct the score-time trajectory by mapping x_pred through the
        # piece's own empirical x -> onset-frame relation, which we can recover
        # from the (x_gt, gt-frame) pairs the dump already carries.
        xg, xp = np.asarray(d['x_gt'], float), np.asarray(d['x_pred'], float)
        fr = np.asarray(d['frame'], float)
        o = np.argsort(xg)
        c2o_pred = np.interp(xp, xg[o], fr[o])
        c2o_gt = np.interp(xg, xg[o], fr[o])
        # sanity: the reconstructed map must reproduce the harness's own error
        c2o_check.append(np.abs(np.abs(c2o_pred - c2o_gt) - np.asarray(d['err_frames'], float)))
        rf = brazier_reliability(c2o_pred)

        d = dict(d)
        d['brazier_unreliable'] = (~rf).astype(float)

        # silence trigger
        wav = find_wav(args.audio_root, page, args.audio_suffix)
        if wav:
            db = frame_rms_db(wav, int(fr.max()) + 1, 1102, 2048, 22050)
            d['silence_db'] = -db[np.asarray(d['frame'], int)]      # higher = quieter = "lost"
        else:
            missing_audio.append(page)
            d['silence_db'] = np.zeros(n)

        for k, v in d.items():
            cols.setdefault(k, []).append(np.asarray(v, float))
        pieces += [piece_of(page)] * n
        page_of_row += [page] * n
        onset_mask += list(np.asarray(d['is_onset'], bool))

    T = {k: np.concatenate(v) for k, v in cols.items()}
    pieces = np.array(pieces)
    onset_mask = np.array(onset_mask, bool)
    err_s = T['err_frames'] / fps
    y_all = err_s > args.threshold_s

    R['meta']['n_frames_total'] = int(len(err_s))
    R['meta']['n_frames_onset'] = int(onset_mask.sum())
    R['meta']['pct_at_0.5s_onsets'] = round(100 * float(1 - y_all[onset_mask].mean()), 2)
    R['meta']['lost_rate_onsets'] = round(float(y_all[onset_mask].mean()), 4)
    R['meta']['available_signals'] = sorted(T)
    chk = np.concatenate(c2o_check)
    R['meta']['c2o_reconstruction_median_abs_err_frames'] = round(float(np.median(chk)), 4)
    R['meta']['c2o_reconstruction_p95_abs_err_frames'] = round(float(np.percentile(chk, 95)), 4)
    R['meta']['pages_without_audio'] = missing_audio

    # ---- signal registry: name -> (array, polarity) where +1 means higher=lost
    reg = {}
    for name, pol in [('conf1', -1), ('conf_max', -1), ('conf_mass', -1),
                      ('margin', -1), ('ent_x', +1), ('silence_db', +1)]:
        if name in T:
            reg[name] = pol * T[name]
    if 'conf1' in T and 'conf2' in T:
        reg['top2_margin'] = -(T['conf1'] - T['conf2'])
    reg['brazier_rf_binary'] = T['brazier_unreliable']

    model_feats = [k for k in ('conf1', 'conf2', 'conf_max', 'conf_mass', 'margin', 'ent_x')
                   if k in T]

    for subset, mask in (('onsets', onset_mask), ('all_frames', np.ones(len(err_s), bool))):
        y = y_all[mask]
        pg = pieces[mask]
        res = {}
        for name, s in reg.items():
            sv = s[mask]
            res[name] = {'auroc': round(auroc(sv, y), 4),
                         'auroc_ci95': cluster_ci(auroc, sv, y, pg, B=args.B),
                         'average_precision': round(average_precision(sv, y), 4),
                         'base_rate': round(float(y.mean()), 4)}
        # binary operating points of the two published rules
        res['brazier_rf_binary'].update(binary_stats(T['brazier_unreliable'][mask] > 0.5, y))
        if 'silence_db' in T:
            res['silence_-40dB'] = binary_stats(-T['silence_db'][mask] < -40, y)

        # trained combination, leave-one-piece-out
        if model_feats:
            X = np.stack([T[f][mask] for f in model_feats], 1)
            oof = logistic_loo(X, y.astype(float), pg)
            e, diag = ece(oof, y)
            res['combo_loo_logreg'] = {
                'features': model_feats,
                'auroc': round(auroc(oof, y), 4),
                'auroc_ci95': cluster_ci(auroc, oof, y, pg, B=args.B),
                'average_precision': round(average_precision(oof, y), 4),
                'ece_10bin': round(e, 4),
                'reliability_diagram': diag}
        R[subset] = res

    # ---- head-to-head vs the published heuristics, paired over pieces -------
    best = max((k for k in R['onsets'] if 'auroc' in R['onsets'][k] and k != 'brazier_rf_binary'),
               key=lambda k: R['onsets'][k]['auroc'])
    R['headline'] = {
        'best_continuous_signal': best,
        'best_auroc': R['onsets'][best]['auroc'],
        'best_auroc_ci95': R['onsets'][best]['auroc_ci95'],
        'brazier_rf_auroc': R['onsets']['brazier_rf_binary']['auroc'],
        'brazier_rf_balanced_accuracy': R['onsets']['brazier_rf_binary'].get('balanced_accuracy'),
        'silence_auroc': R['onsets'].get('silence_db', {}).get('auroc'),
    }
    if 'combo_loo_logreg' in R['onsets']:
        R['headline']['combo_auroc'] = R['onsets']['combo_loo_logreg']['auroc']
        R['headline']['combo_auroc_ci95'] = R['onsets']['combo_loo_logreg']['auroc_ci95']

    # paired bootstrap on the AUROC gap vs each published heuristic
    for rival in ('brazier_rf_binary', 'silence_db'):
        if rival not in R['onsets']:
            continue
        a = reg[best][onset_mask]
        b = reg[rival][onset_mask]
        y = y_all[onset_mask]
        pg = pieces[onset_mask]
        rng = np.random.default_rng(11)
        uniq = sorted(set(pg))
        idx = {p: np.flatnonzero(pg == p) for p in uniq}
        gaps = []
        for _ in range(args.B):
            take = np.concatenate([idx[uniq[j]] for j in rng.integers(0, len(uniq), len(uniq))])
            g = auroc(a[take], y[take]) - auroc(b[take], y[take])
            if np.isfinite(g):
                gaps.append(g)
        R['headline'][f'gap_vs_{rival}'] = {
            'delta_auroc': round(auroc(a, y) - auroc(b, y), 4),
            'ci95': [round(float(np.percentile(gaps, 2.5)), 4),
                     round(float(np.percentile(gaps, 97.5)), 4)],
            'significant': bool(np.percentile(gaps, 2.5) > 0 or np.percentile(gaps, 97.5) < 0)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(R, open(args.out, 'w'), indent=2)
    print(f'wrote {args.out}')
    print(f"\npages={R['meta']['n_pages']}  frames={R['meta']['n_frames_total']} "
          f"onsets={R['meta']['n_frames_onset']}  pct@0.5s={R['meta']['pct_at_0.5s_onsets']} "
          f"lost_rate={R['meta']['lost_rate_onsets']}")
    print('\nAUROC of "this frame is lost (err>0.5s)", onset frames:')
    for k, v in sorted(R['onsets'].items(), key=lambda kv: -(kv[1].get('auroc') or 0)):
        if 'auroc' in v:
            print(f"  {k:22s} AUROC={v['auroc']:.3f}  CI{v['auroc_ci95']}  AP={v['average_precision']:.3f}")
    for k in ('brazier_rf_binary', 'silence_-40dB'):
        if k in R['onsets'] and 'balanced_accuracy' in R['onsets'][k]:
            v = R['onsets'][k]
            print(f"  [binary] {k:20s} bal.acc={v['balanced_accuracy']:.3f} "
                  f"TPR={v['tpr_recall_of_lost']:.3f} FPR={v['fpr']:.3f} flag_rate={v['flag_rate']:.3f}")
    print('\nheadline:', json.dumps(R['headline'], indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
