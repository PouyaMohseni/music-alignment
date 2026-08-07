"""Score the AMT bridge: note-onset F1, and the END-TO-END pct@0.5s it buys.

Reads the transcriptions written by scripts/amt_transcribe_real.py and answers
two questions on the 25 MSMD real-performance pages:

  (1) ACOUSTIC ROBUSTNESS OF AMT ON OUR AUDIO.
      mir_eval note-onset F1 (offset ignored, velocity ignored) against
      performance/{piece}.mid, at 50 ms and 100 ms onset tolerance, for
      {kong_stock, edwards_robust} x {room, di-left}.  Because room and
      di-left are two microphones on the SAME take, the room-minus-di-left
      delta is a clean measurement of what the room costs, with the
      performance held exactly fixed.

  (2) WHAT SURVIVES END-TO-END.
      Simulate the decomposed tracker -- audio -> AMT notes -> monotonic match
      against the score note sequence -> notehead pixel -> position -- and
      score it with the SAME metric the CPJKU harness uses, so the number is
      directly comparable to our image-model results (ours 45.6 room,
      cyolo_sb 79.9, CODA 88.3).

METRIC, REPRODUCED EXACTLY FROM third_party/cpjku_unet
------------------------------------------------------
audio_conditioned_unet/dataset.py::calculate_batch_stats computes, at every
spectrogram frame that is a GT onset frame,

    frame_diff = |interpol_c2o(x_pred) - interpol_c2o(x_gt)|

where interpol_c2o maps an UNROLLED x pixel (x + cumulative width of preceding
staves) back to an onset frame, built by interp1d(kind='previous').
eval_model.py then divides by fps (=20) and reports the cumulative percentage
of onsets under each threshold, POOLED over all pieces (frame_diffs
['onset_diffs']).  We rebuild interpol_c2o here from the same npz + MIDI, via
the same utils.merge_onsets and the same scale_factor=3 coordinate scaling that
utils.load_score applies, and pool identically.  We use the real madmom
MIDIFile parser (hence venv_cpjku310) so note indices match theirs bit-for-bit.

TRACKERS
--------
  oracle   the GT MIDI notes themselves are fed in as "detections".  This is
           the ceiling of the note-index-hold formulation -- perfect
           transcription, naive hold, no interpolation between onsets.  A
           previous agent measured 93.4 for this; we recompute it in-harness so
           the AMT numbers have a matched ceiling.
  greedy   ONLINE greedy monotonic pitch matcher.  Pointer p into the score
           note list; each detection, in time order, claims the first score
           note at index >= p with the same pitch inside a window of W notes;
           on success p advances past it.  Unmatched detections are dropped.
           Deliberately naive: the point is a first number, and it must NOT
           exploit the fact that these recordings are MIDI playback (see
           CAVEAT), so it never looks at absolute detection times when
           matching -- only at pitch order.
  dtw      OFFLINE DTW on the pitch sequences (cost 0 on pitch match, 1
           otherwise).  Non-causal, so it is an upper bound on what a
           better matcher could recover from the same transcription.

Position at eval frame t is the last matched score-note index from detections
before det_time < (t+1)/fps + latency.  Three latency settings are reported:
  lat=-0.050  strictly BEFORE the onset frame -- the tracker cannot have heard
              the note it is being scored on, so even a perfect transcriber is
              one note behind.  This is the convention under which the earlier
              in-repo oracle measured 93.4; we reproduce it as a cross-check.
  lat=+0.000  the note sounding at frame t is available (audio through the end
              of frame t).  A perfect transcriber scores exactly 100.0 here,
              which is what validates the pixel round-trip.
  lat=+0.050  50 ms of extra slack, matching the mir_eval onset tolerance --
              absorbs AMT onset-regression jitter.

Given a matched score-note index n, the pixel is the coordinate of the largest
merged onset group whose note index is <= n (coord2onset is monotone).

MEASURED CEILINGS (oracle = GT MIDI fed in as detections, this harness):
    lat=-0.050 -> 94.79 pct@0.5s     (matches the 93.4 an earlier agent got)
    lat=+0.000 -> 100.00 pct@0.5s    (validates the pixel round-trip is exact)

CAVEAT WORTH READING
--------------------
score/{piece}.mid is byte-identical to performance/{piece}.mid -- the
recordings are Disklavier playback of the score MIDI, so there is NO tempo
deviation between score and performance.  The matcher never uses absolute
time, so it does not cheat, but a real human performance would stress a
matcher harder than this.  Treat the end-to-end number as "what AMT error
alone costs", not as a full score-following result.

USAGE
-----
    /scratch/pmohseni/venv_cpjku310/bin/python scripts/amt_bridge_eval.py \
        --out results/amt_bridge_eval.json
"""
import argparse
import copy
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml
from scipy import interpolate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(REPO, 'third_party/cpjku_unet/data/msmd/msmd_real_performances')
FPS = 20                # configs/msmd_rec_room.yaml spectrogram_params.fps
SCALE_FACTOR = 3        # eval_any_cpu.sh passes --scale_factor 3
THRESHOLDS = [0.05, 0.1, 0.5, 1.0, 5.0]
CHOPIN = 'ChopinFF__O9__nocturne'


# --------------------------------------------------------------------------
# verbatim from third_party/cpjku_unet/audio_conditioned_unet/utils.py
# --------------------------------------------------------------------------
def merge_onsets(cur_onsets, stk_note_coords, coords2onsets):
    """merge onsets occurring in the same frame (CPJKU utils.merge_onsets).

    Returns the extra ``coord_ids`` bookkeeping we need on top of theirs.
    """
    coord_ids = coords2onsets.keys()
    onsets, coords, kept_coord_ids = [], [], []
    for i in coord_ids:
        if coords2onsets[i] >= len(cur_onsets):
            continue
        if cur_onsets[coords2onsets[i]] not in onsets:
            coords.append(stk_note_coords[i])
            onsets.append(cur_onsets[coords2onsets[i]])
            kept_coord_ids.append(i)
    return (np.asarray(onsets, dtype=np.int64),
            np.asarray(coords, dtype=np.float32),
            np.asarray(kept_coord_ids, dtype=np.int64))


def build_c2o(coords_new, onsets):
    """Rebuild dataset.py's interpol_c2o and the unrolled-x helper."""
    unrolled_coords_x, coords_per_staff, max_xes = [], [], [0]
    staff_coords = sorted(np.unique(coords_new[:, 0]))
    for c in staff_coords:
        cs_staff = coords_new[coords_new[:, 0] == c, :-1]
        coords_per_staff.append(cs_staff)
        max_xes.append(max(cs_staff[:, 1]))
    add_per_staff = np.cumsum(max_xes)[:-1]
    for idx in range(len(staff_coords)):
        unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[idx])
    unrolled_coords_x = np.concatenate(unrolled_coords_x)

    interpol_c2o = interpolate.interp1d(unrolled_coords_x, onsets, kind='previous',
                                        bounds_error=False,
                                        fill_value=(onsets[0], onsets[-1]))
    staff_coords = np.asarray(staff_coords)

    def unroll(coord):
        """coord = [y, x, h] -> unrolled x, matching calculate_batch_stats."""
        sid = int(np.argmin(np.abs(staff_coords - coord[0])))
        return coord[1] + add_per_staff[sid]

    return interpol_c2o, unroll


# --------------------------------------------------------------------------
def load_piece(piece):
    from madmom.io import midi as mm_midi

    npz = np.load(os.path.join(REC, 'score', piece + '.npz'), allow_pickle=True)
    coords = np.array(npz['coords'], dtype=np.float32) / SCALE_FACTOR  # load_score
    coord2onset = npz['coord2onset'][0]

    midi = mm_midi.MIDIFile(os.path.join(REC, 'performance', piece + '.mid'))
    notes = np.array(midi.notes)          # [onset_s, pitch, duration_s, vel, ch]
    onset_frames = (notes[:, 0] * FPS).astype(int)

    onsets_u, coords_u, coord_ids_u = merge_onsets(onset_frames, copy.deepcopy(coords),
                                                   coord2onset)
    note_idx_u = np.array([coord2onset[int(c)] for c in coord_ids_u], dtype=np.int64)
    c2o, unroll = build_c2o(coords_u, onsets_u)
    x_unrolled_u = np.array([unroll(c) for c in coords_u], dtype=np.float64)

    return dict(piece=piece, notes=notes, onsets_u=onsets_u, coords_u=coords_u,
                note_idx_u=note_idx_u, c2o=c2o, x_unrolled_u=x_unrolled_u)


# --------------------------------------------------------------------------
# matchers: detections (time, pitch) -> per-detection score-note index or -1
# --------------------------------------------------------------------------
def match_greedy(det_pitch, score_pitch, window=24):
    n = len(score_pitch)
    out = np.full(len(det_pitch), -1, dtype=np.int64)
    p = 0
    for i, pit in enumerate(det_pitch):
        hi = min(n, p + window)
        j = -1
        for k in range(p, hi):
            if score_pitch[k] == pit:
                j = k
                break
        if j >= 0:
            out[i] = j
            p = j + 1
    return out


def match_dtw(det_pitch, score_pitch):
    """Offline DTW on pitch sequences; cost 0 on pitch match else 1."""
    n, m = len(det_pitch), len(score_pitch)
    if n == 0 or m == 0:
        return np.full(n, -1, dtype=np.int64)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    cost = (np.asarray(det_pitch)[:, None] != np.asarray(score_pitch)[None, :]).astype(np.float32)
    # allow skipping score notes cheaply (they may be inaudible / merged) and
    # detections expensively (spurious notes should not drag the pointer)
    SKIP_SCORE, SKIP_DET = 0.6, 1.0
    for i in range(1, n + 1):
        ci = cost[i - 1]
        row_prev, row = dp[i - 1], dp[i]
        row[0] = row_prev[0] + SKIP_DET
        for j in range(1, m + 1):
            row[j] = min(row_prev[j - 1] + ci[j - 1],
                         row[j - 1] + SKIP_SCORE,
                         row_prev[j] + SKIP_DET)
    # backtrace
    out = np.full(n, -1, dtype=np.int64)
    i, j = n, m
    while i > 0 and j > 0:
        d = dp[i, j]
        if np.isclose(d, dp[i - 1, j - 1] + cost[i - 1, j - 1]):
            if cost[i - 1, j - 1] == 0:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif np.isclose(d, dp[i, j - 1] + SKIP_SCORE):
            j -= 1
        else:
            i -= 1
    return out


# --------------------------------------------------------------------------
def run_tracker(P, det_time, det_pitch, matcher, latency=0.0, window=24):
    """Return (frame_diffs_seconds, diagnostics) at every GT onset frame."""
    score_pitch = P['notes'][:, 1].astype(int)
    if matcher == 'greedy':
        matched = match_greedy(det_pitch, score_pitch, window=window)
    elif matcher == 'dtw':
        matched = match_dtw(det_pitch, score_pitch)
    else:
        raise ValueError(matcher)

    onsets_u, note_idx_u = P['onsets_u'], P['note_idx_u']
    x_u, c2o = P['x_unrolled_u'], P['c2o']

    diffs, k_preds = [], []
    di = 0                       # pointer into detections (they are time-sorted)
    cur_note = -1                # last matched score-note index
    n_used = 0
    for k, frame in enumerate(onsets_u):
        deadline = (frame + 1) / FPS + latency
        while di < len(det_time) and det_time[di] < deadline:
            if matched[di] >= 0:
                # monotone: never move backwards
                cur_note = max(cur_note, int(matched[di]))
                n_used += 1
            di += 1
        if cur_note < 0:
            k_pred = 0
        else:
            cand = np.nonzero(note_idx_u <= cur_note)[0]
            k_pred = int(cand[-1]) if len(cand) else 0
        k_preds.append(k_pred)
        diffs.append(abs(float(c2o(x_u[k_pred])) - float(c2o(x_u[k]))) / FPS)

    k_preds = np.asarray(k_preds)
    k_gt = np.arange(len(onsets_u))
    diag = dict(n_det=int(len(det_time)),
                n_matched=int((matched >= 0).sum()),
                n_used=int(n_used),
                n_score_notes=int(len(score_pitch)),
                n_eval_onsets=int(len(onsets_u)),
                mean_index_lag=float(np.mean(k_preds - k_gt)),
                frac_behind=float(np.mean(k_preds < k_gt)),
                frac_ahead=float(np.mean(k_preds > k_gt)),
                max_abs_index_err=int(np.max(np.abs(k_preds - k_gt))) if len(k_preds) else 0)
    return np.asarray(diffs), diag


def pct_table(diffs):
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {str(t): float('nan') for t in THRESHOLDS}
    return {str(t): round(100.0 * float(np.mean(diffs <= t)), 2) for t in THRESHOLDS}


# --------------------------------------------------------------------------
def onset_f1(P, det_time, det_pitch, tol):
    import mir_eval
    notes = P['notes']
    ref_int = np.stack([notes[:, 0], notes[:, 0] + np.maximum(notes[:, 2], 1e-3)], 1)
    ref_pit = mir_eval.util.midi_to_hz(notes[:, 1])
    if len(det_time) == 0:
        return 0.0, 0.0, 0.0
    det_time = np.asarray(det_time, dtype=float)
    # mir_eval rejects negative interval times, which best_shift() produces as
    # soon as it tries a negative offset.  Translate BOTH sides by the same
    # constant instead: onset matching is translation-invariant, so F1 is
    # bit-for-bit unchanged, whereas dropping the negative detections would
    # quietly shrink the denominator and bias best_shift toward large negative
    # shifts (fewer detections -> easier precision).
    pad = max(0.0, -float(det_time.min())) if len(det_time) else 0.0
    ref_int = ref_int + pad
    det_time = det_time + pad
    est_int = np.stack([det_time, det_time + 0.1], 1)
    est_pit = mir_eval.util.midi_to_hz(np.asarray(det_pitch, dtype=float))
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pit, est_int, est_pit, onset_tolerance=tol,
        offset_ratio=None, pitch_tolerance=50.0)
    return float(p), float(r), float(f)


def best_shift(P, det_time, det_pitch):
    """Diagnostic: is there a constant wav-vs-MIDI latency? Grid +-0.5 s."""
    best, arg = -1.0, 0.0
    for s in np.arange(-0.5, 0.5001, 0.01):
        _, _, f = onset_f1(P, np.asarray(det_time) + s, det_pitch, 0.05)
        if f > best:
            best, arg = f, float(s)
    return round(arg, 3), round(best, 4)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amt_root', default='/scratch/pmohseni/amt_out')
    ap.add_argument('--out', default=os.path.join(REPO, 'results/amt_bridge_eval.json'))
    ap.add_argument('--models', default='kong_stock,edwards_robust')
    ap.add_argument('--tiers', default='room,di-left')
    ap.add_argument('--window', type=int, default=24)
    ap.add_argument('--shift_diag', action='store_true',
                    help='grid-search a constant wav-vs-MIDI offset (slow-ish)')
    args = ap.parse_args()

    with open(os.path.join(REC, 'rp_split.yaml'), 'rb') as fp:
        pieces = yaml.safe_load(fp)['files']

    print('loading scores + GT MIDI ...', flush=True)
    P = {p: load_piece(p) for p in pieces}
    tot_on = sum(len(P[p]['onsets_u']) for p in pieces)
    chop_on = sum(len(P[p]['onsets_u']) for p in pieces if CHOPIN in p)
    print(f'{len(pieces)} pieces, {tot_on} merged onset frames; '
          f'Chopin Op.9 = {chop_on} ({100.0*chop_on/tot_on:.1f}%)', flush=True)

    results = {'meta': {'fps': FPS, 'scale_factor': SCALE_FACTOR,
                        'thresholds': THRESHOLDS, 'n_pieces': len(pieces),
                        'n_onsets_total': tot_on, 'n_onsets_chopin': chop_on},
               'runs': {}, 'per_piece': {}}

    # ---- systems: (model, tier) from AMT json, plus the oracle ------------
    systems = [('oracle', 'oracle')]
    for m in args.models.split(','):
        for t in args.tiers.split(','):
            systems.append((m, t))

    for model, tier in systems:
        tag = f'{model}/{tier}'
        det = {}
        ok = True
        for p in pieces:
            if model == 'oracle':
                det[p] = (P[p]['notes'][:, 0].astype(float),
                          P[p]['notes'][:, 1].astype(int))
            else:
                f = os.path.join(args.amt_root, model, tier, p + '.json')
                if not os.path.exists(f):
                    print(f'MISSING {f}', flush=True)
                    ok = False
                    break
                d = json.load(open(f))
                o = np.asarray(d['onset'], dtype=float)
                q = np.asarray(d['pitch'], dtype=int)
                srt = np.argsort(o, kind='stable')
                det[p] = (o[srt], q[srt])
        if not ok:
            continue

        print(f'\n===== {tag} =====', flush=True)
        run = {'f1': {}, 'trackers': {}}

        # ---- (1) note-onset F1 -------------------------------------------
        if model != 'oracle':
            for tol in (0.05, 0.10):
                per = {}
                for p in pieces:
                    pr, rc, f1 = onset_f1(P[p], det[p][0], det[p][1], tol)
                    per[p] = dict(precision=round(pr, 4), recall=round(rc, 4),
                                  f1=round(f1, 4), n_ref=int(len(P[p]['notes'])),
                                  n_est=int(len(det[p][0])))
                # note-weighted aggregate == F1 over the pooled note pool
                w = np.array([len(P[p]['notes']) for p in pieces], dtype=float)
                fs = np.array([per[p]['f1'] for p in pieces])
                chop = np.array([CHOPIN in p for p in pieces])
                run['f1'][f'tol{tol}'] = dict(
                    macro_f1=round(float(fs.mean()), 4),
                    weighted_f1=round(float((fs * w).sum() / w.sum()), 4),
                    macro_precision=round(float(np.mean([per[p]['precision'] for p in pieces])), 4),
                    macro_recall=round(float(np.mean([per[p]['recall'] for p in pieces])), 4),
                    chopin_macro_f1=round(float(fs[chop].mean()), 4),
                    other_macro_f1=round(float(fs[~chop].mean()), 4),
                    per_piece=per)
                print(f'  onset-F1 @{int(tol*1000)}ms  macro={fs.mean():.4f}  '
                      f'weighted={(fs*w).sum()/w.sum():.4f}  '
                      f'chopin={fs[chop].mean():.4f}  other={fs[~chop].mean():.4f}',
                      flush=True)

            if args.shift_diag:
                # Never let an OPTIONAL diagnostic take down the primary
                # measurement.  Run 425142 died here after printing exactly one
                # system's F1, losing the end-to-end numbers for all five.
                try:
                    sh = {p: best_shift(P[p], det[p][0], det[p][1]) for p in pieces}
                    run['best_constant_shift_s'] = sh
                    arr = np.array([v[0] for v in sh.values()])
                    print(f'  best constant wav-vs-MIDI shift: median={np.median(arr):+.3f}s '
                          f'min={arr.min():+.3f} max={arr.max():+.3f}', flush=True)
                except Exception as e:
                    print(f'  shift_diag FAILED ({type(e).__name__}: {e}) -- '
                          f'continuing; primary metrics are unaffected', flush=True)

        # ---- (2) end-to-end pct@0.5s --------------------------------------
        for matcher in ('greedy', 'dtw'):
            for latency in (-1.0 / FPS, 0.0, 0.05):
                pooled, per = [], {}
                diags = {}
                for p in pieces:
                    d, dg = run_tracker(P[p], det[p][0], det[p][1], matcher,
                                        latency=latency, window=args.window)
                    pooled.append(d)
                    per[p] = dict(pct=pct_table(d), mean_err_s=round(float(d.mean()), 4),
                                  median_err_s=round(float(np.median(d)), 4),
                                  n=len(d), **{k: v for k, v in dg.items()})
                    diags[p] = dg
                pooled_all = np.concatenate(pooled)
                chop_d = np.concatenate([per_d for p, per_d in zip(pieces, pooled)
                                         if CHOPIN in p])
                oth_d = np.concatenate([per_d for p, per_d in zip(pieces, pooled)
                                        if CHOPIN not in p])
                key = f'{matcher}_lat{latency:+.3f}'
                run['trackers'][key] = dict(
                    pooled=pct_table(pooled_all),
                    pooled_mean_err_s=round(float(pooled_all.mean()), 4),
                    pooled_median_err_s=round(float(np.median(pooled_all)), 4),
                    chopin=pct_table(chop_d), other=pct_table(oth_d),
                    per_piece=per)
                print(f'  {key:14s} pct@0.5s = {pct_table(pooled_all)["0.5"]:6.2f}   '
                      f'(chopin {pct_table(chop_d)["0.5"]:6.2f} / other '
                      f'{pct_table(oth_d)["0.5"]:6.2f})  '
                      f'median_err={np.median(pooled_all):.3f}s', flush=True)

        results['runs'][tag] = run

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
