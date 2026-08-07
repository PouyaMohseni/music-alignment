#!/usr/bin/env python
"""Score oemer's notehead output against MSMD's LilyPond ground truth.

Everything is computed in *native MSMD page pixels* (835x1181).  For the
`hires` variant oemer's coordinates are divided by the render scale first, so
the two variants' errors are directly comparable.

Scale constants, all MEASURED on these pages, not assumed:
    staff space   7.0 px  (identical on every page)
    notehead      10 x 9 px  (MUNG bbox, identical on every page)
    GT noise floor 0.75 px median  (MUNG bbox centre vs MSMD's own
                  notes_NN.npy coordinate for the same notehead)

Two ground-truth sets:
    ALL      every notehead on the page, from mung/NN.xml.  Primary --
             scoring detection against the aligned subset alone would charge
             oemer a false positive for correctly finding the far side of a
             tie.
    ALIGNED  coords/notes_NN.npy, the subset MSMD tied to a MIDI onset.  This
             is what our score-following pipeline actually consumes, so its
             recall is reported separately.

Staff-line position replicates oemer's own convention exactly
(oemer/notehead_extraction.py:341-367): 0 == half-space below the bottom
line (D4 read in treble), 1 == bottom line, 9 == top line, up is positive.
GT staff lines are read off the page image with a comb matched-filter, so
both sides of the comparison are geometric.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

STAFF_SPACE = 7.0
NH_W = 10.0
NH_H = 9.0
SEMI = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
# oemer staff_line_pos 0 in each clef, as a diatonic index (7*octave + step)
CLEF_ZERO = {"G": 7 * 4 + 1, "F": 7 * 2 + 3, "C": 7 * 3 + 2}  # D4, F2, E3


def nat_midi(dia):
    return 12 * (dia // 7 + 1) + SEMI[dia % 7]


# ----------------------------------------------------------------- GT staves
def gt_staves(img_path, systems, space=STAFF_SPACE, resp_th=3.6):
    """Recover each staff's 5 line y-positions from the page image itself.

    A comb matched-filter rather than "find rows that are mostly black, then
    cut on big gaps".  The naive version breaks on real pages two ways: long
    slurs/beams spanning most of a system look exactly like a staff line, and
    a line heavily occluded by noteheads drops below any coverage threshold.
    Sliding a 5-tooth comb of the known staff spacing scores the whole
    structure at once, so a stray horizontal scores ~1/5 and one weak line
    still leaves ~4/5.
    """
    blk = np.array(Image.open(img_path).convert("L")) < 128
    staves = []
    for b in systems:
        y0, y1 = int(b[:, 0].min()), int(b[:, 0].max())
        x0, x1 = int(b[:, 1].min()), int(b[:, 1].max())
        pad = int(round(2 * space))
        ya, yb = max(0, y0 - pad), min(blk.shape[0], y1 + pad + 1)
        prof = blk[ya:yb, x0 : x1 + 1].mean(1)
        prof = np.maximum(prof, np.roll(prof, 1))
        n = len(prof)
        offs = [int(round(k * space)) for k in range(5)]
        resp = np.full(n, -1.0)
        for y in range(0, n - offs[-1]):
            resp[y] = sum(prof[y + o] for o in offs)
        taken = []
        for y in np.argsort(-resp):
            if resp[y] < resp_th:
                break
            if any(abs(y - t) < 3 * space for t in taken):
                continue
            taken.append(int(y))
        for y in sorted(taken):
            # sub-pixel line centres.  argmax over a 3px window quantises to
            # whole pixels, and one staff-position step is only 3.5 px, so a
            # 1.5 px line error is enough to flip a notehead to the next
            # staff position.  Coverage-weighted centroid instead.
            lines = []
            for o in offs:
                a, b = max(0, y + o - 2), min(n, y + o + 3)
                w = prof[a:b].astype(float)
                w = np.clip(w - 0.25, 0, None)
                idx = np.arange(a, b)
                lines.append(float(ya + (idx * w).sum() / w.sum()) if w.sum() > 0
                             else float(ya + y + o))
            staves.append(dict(lines=lines, x0=x0, x1=x1))
    staves.sort(key=lambda s: s["lines"][0])
    for s in staves:
        L = np.sort(np.array(s["lines"]))
        # least squares over all 5 lines rather than trusting the bottom one
        k = np.arange(5)
        A = np.stack([np.ones(5), k], 1)
        c, sp = np.linalg.lstsq(A, L, rcond=None)[0]
        s["lines"] = L.tolist()
        s["y_upper"], s["y_lower"] = float(L.min()), float(L.max())
        s["y_center"] = float(L.mean())
        s["space"] = float(sp)
        s["bottom"] = float(c + 4 * sp)
        s["phase"] = 0.0
    return staves


def calibrate_phase(staves, staff_idx, ys):
    """Remove each staff's residual sub-pixel offset from the half-space grid.

    Noteheads sit on exact half-space positions by construction, so the median
    fractional part of (bottom - y)/step over a staff's own noteheads is pure
    measurement error -- in the line centres and in the MUNG bbox centre's
    integer quantisation.  It is systematically ~0.3 of a step here, which is
    close enough to the 0.5 rounding boundary to flip individual notes.
    Subtracting it cannot change the grid's integer indexing, only de-bias it.
    """
    for si, s in enumerate(staves):
        r = [(s["bottom"] - y) / (s["space"] / 2.0)
             for i, y in enumerate(ys) if staff_idx[i] == si]
        if len(r) < 4:
            continue
        fr = np.mod(np.array(r) + 0.5, 1.0) - 0.5
        s["phase"] = float(np.median(fr))
    return staves


def assign_staff(y, x, staves):
    if not staves:
        return None
    cand = [
        i for i, s in enumerate(staves)
        if s["y_upper"] - 3 * STAFF_SPACE <= y <= s["y_lower"] + 3 * STAFF_SPACE
        and s["x0"] - 20 <= x <= s["x1"] + 20
    ]
    pool = cand if cand else list(range(len(staves)))
    return pool[int(np.argmin([abs(staves[i]["y_center"] - y) for i in pool]))]


def slp_of(y, s):
    return int(round((s["bottom"] - y) / (s["space"] / 2.0) - s.get("phase", 0.0))) + 1


def infer_clefs(gt, staff_idx, slps, staves):
    """Recover each staff's clef from GT alone.

    We have the geometric staff position and the true MIDI pitch of every
    notehead; the clef is whatever constant offset reconciles them.  Solving
    for it gives GT clefs for free and makes oemer's clef itself measurable.

    Only unambiguous in-staff noteheads (0 <= slp <= 10) vote: a ledger note
    between the two staves of a grand staff cannot be attributed to one staff
    geometrically, which is the same ambiguity oemer faces.  The returned
    confidence is the vote share, and the caller drops staves below 0.9 from
    the pitch metric rather than reporting a number built on a guessed clef.
    """
    out = {}
    for si in range(len(staves)):
        idx = [i for i in range(len(gt))
               if staff_idx[i] == si and gt[i, 4] > 0 and 0 <= slps[i] <= 10]
        if not idx:
            out[si] = None
            continue
        best, bestn = None, -1
        for name, zero in CLEF_ZERO.items():
            n = sum(abs(nat_midi(zero + slps[i]) - int(gt[i, 4])) <= 1 for i in idx)
            if n > bestn:
                best, bestn = name, n
        out[si] = (best, bestn / len(idx))
    return out


# ----------------------------------------------------------------- matching
def match(gt, det, tol):
    """Optimal 1-1 assignment under a Euclidean gate of `tol` px.

    Hungarian rather than greedy: inside a dense chord, greedy nearest-first
    can lock a detection onto the wrong notehead and cascade down the stack.
    """
    if len(gt) == 0 or len(det) == 0:
        return np.zeros((0, 2), int)
    d = np.linalg.norm(gt[:, None, :] - det[None, :, :], axis=2)
    BIG = 1e6
    cost = np.where(d <= tol, d, BIG)
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < BIG
    return np.stack([r[keep], c[keep]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/scratch/pmohseni/omr")
    ap.add_argument("--variant", default="native")
    ap.add_argument("--tol", type=float, default=NH_W / 2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-invalid", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    man = {m["key"]: m for m in json.load(open(args.work + "/pages/manifest.json"))}
    outdir = os.path.join(args.work, "out", args.variant)
    TOLS = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0]

    pages, per_note = [], []
    sens = defaultdict(lambda: [0, 0, 0])

    for f in sorted(glob.glob(outdir + "/*.json")):
        rec = json.load(open(f))
        key = os.path.basename(f)[:-5]
        m = man[key]
        S = m["hires_scale"] if args.variant == "hires" else 1

        mg = np.load(args.work + "/pages/gt/" + key + "_mung.npz", allow_pickle=True)
        g = mg["nh"]                       # (N,6) x,y,w,h,pitch,tied
        gt = g[:, :2]                      # (x,y) page px
        aligned = np.load(m["gt"])["notes"]  # (M,2) [y,x]
        aligned_xy = np.stack([aligned[:, 1], aligned[:, 0]], 1)
        systems = np.load(m["gt"])["systems"]

        notes = rec["notes"]
        if args.drop_invalid:
            notes = [n for n in notes if not n["invalid"]]
        det = np.array([[n["cx"] / S, n["cy"] / S] for n in notes]).reshape(-1, 2)

        pairs = match(gt, det, args.tol)
        tp = len(pairs)
        prec = tp / len(det) if len(det) else 0.0
        rc = tp / len(gt) if len(gt) else 0.0
        f1 = 2 * prec * rc / (prec + rc) if prec + rc else 0.0
        al_pairs = match(aligned_xy, det, args.tol)
        rc_aligned = len(al_pairs) / len(aligned_xy) if len(aligned_xy) else 0.0

        for t in TOLS:
            p2 = match(gt, det, t)
            s = sens[t]
            s[0] += len(p2); s[1] += len(gt); s[2] += len(det)

        # ---- GT geometry: staves, staff-line positions, clefs ----
        stv = gt_staves(m["src_img"], systems)
        sidx = [assign_staff(g[i, 1], g[i, 0], stv) for i in range(len(g))]
        stv = calibrate_phase(stv, sidx, g[:, 1])
        gslp = [slp_of(g[i, 1], stv[sidx[i]]) if sidx[i] is not None else None
                for i in range(len(g))]
        clefs = infer_clefs(g, sidx, gslp, stv)

        # oemer's clef for each of its (group, track) staves, if it got that far
        ocl = {}
        for c in rec.get("clefs", []):
            lab = (c["label"] or "").replace("ClefType.", "").split("_")[0]
            ocl.setdefault((c["group"], c["track"]), []).append((c["cx"] / S, lab))
        for k in ocl:
            ocl[k].sort()

        dx = det[pairs[:, 1], 0] - gt[pairs[:, 0], 0] if tp else np.array([])
        dy = det[pairs[:, 1], 1] - gt[pairs[:, 0], 1] if tp else np.array([])

        gi2di = {int(a): int(b) for a, b in pairs}
        for gi in range(len(gt)):
            near = np.linalg.norm(gt - gt[gi], axis=1)
            chord = int(np.sum((np.abs(gt[:, 0] - gt[gi, 0]) < 0.6 * NH_W)
                               & (np.abs(gt[:, 1] - gt[gi, 1]) < 4 * STAFF_SPACE)) - 1)
            dens = int(np.sum(near < 3 * STAFF_SPACE) - 1)
            gp = gslp[gi]
            row = dict(key=key, gi=gi, matched=gi in gi2di,
                       gt_x=float(gt[gi, 0]), gt_y=float(gt[gi, 1]),
                       cls=str(mg["cls"][gi]), pitch=int(g[gi, 4]),
                       tied=int(g[gi, 5]), gt_slp=gp, gt_staff=sidx[gi],
                       gt_clef=clefs.get(sidx[gi], (None, 0))[0]
                       if clefs.get(sidx[gi]) else None,
                       chord=chord, dens=dens,
                       ledger=bool(gp is not None and (gp < 0 or gp > 10)))
            if gi in gi2di:
                n = notes[gi2di[gi]]
                row["dx"] = float(det[gi2di[gi], 0] - gt[gi, 0])
                row["dy"] = float(det[gi2di[gi], 1] - gt[gi, 1])
                row["det_slp"] = n["staff_line_pos"]
                row["invalid"] = n["invalid"]
                row["label"] = n["label"]
                # pitch up to accidental: oemer's staff position + its clef
                cl = ocl.get((n["group"], n["track"]))
                dc = None
                if cl:
                    cand = [lab for cx, lab in cl if cx <= n["cx"] / S]
                    dc = cand[-1] if cand else cl[0][1]
                row["det_clef"] = dc
                gc = clefs.get(sidx[gi])
                # only score pitch where the GT clef itself is unambiguous
                if gc and gc[1] >= 0.9 and dc in CLEF_ZERO \
                        and n["staff_line_pos"] is not None:
                    d_ = CLEF_ZERO[dc] + int(round(n["staff_line_pos"]))
                    row["pitch_ok"] = bool(0 <= d_ < 90
                                           and abs(nat_midi(d_) - int(g[gi, 4])) <= 1)
                    row["det_dia"] = d_
            per_note.append(row)

        sl = [r for r in per_note if r["key"] == key and r["matched"]
              and r.get("det_slp") is not None and r["gt_slp"] is not None]
        pk = [r for r in per_note if r["key"] == key and "pitch_ok" in r]
        pages.append(dict(
            key=key, piece=m["piece"], page=m["page"],
            n_gt=len(gt), n_gt_aligned=len(aligned_xy), n_tied=int(g[:, 5].sum()),
            n_det=len(det), n_invalid=sum(1 for n in rec["notes"] if n["invalid"]),
            tp=tp, precision=prec, recall=rc, f1=f1, recall_aligned=rc_aligned,
            med_adx=float(np.median(np.abs(dx))) if tp else None,
            p90_adx=float(np.percentile(np.abs(dx), 90)) if tp else None,
            max_adx=float(np.max(np.abs(dx))) if tp else None,
            med_ady=float(np.median(np.abs(dy))) if tp else None,
            p90_ady=float(np.percentile(np.abs(dy), 90)) if tp else None,
            bias_dx=float(np.mean(dx)) if tp else None,
            bias_dy=float(np.mean(dy)) if tp else None,
            slp_acc=float(np.mean([round(r["det_slp"]) == r["gt_slp"] for r in sl]))
            if sl else None,
            slp_n=len(sl),
            pitch_acc=float(np.mean([r["pitch_ok"] for r in pk])) if pk else None,
            pitch_n=len(pk),
            n_gt_staves=len(stv), n_det_staves=len(rec.get("staffs", [])),
            n_sys=int(len(systems)),
            wall_noteheads=rec.get("wall_to_noteheads"),
            wall_total=rec.get("wall_total"), stages=rec.get("stages"),
            n_errors=len(rec.get("errors", [])),
        ))

    # ---------------- aggregate ----------------
    TP = sum(p["tp"] for p in pages); NG = sum(p["n_gt"] for p in pages)
    ND = sum(p["n_det"] for p in pages)
    P = TP / ND if ND else 0; R = TP / NG if NG else 0
    mt = [r for r in per_note if r["matched"]]
    adx = np.abs([r["dx"] for r in mt]); ady = np.abs([r["dy"] for r in mt])
    sl = [r for r in mt if r.get("det_slp") is not None and r["gt_slp"] is not None]
    pk = [r for r in per_note if "pitch_ok" in r]
    agg = dict(
        variant=args.variant, tol_px=args.tol, drop_invalid=args.drop_invalid,
        n_pages=len(pages), n_gt_all=NG,
        n_gt_aligned=sum(p["n_gt_aligned"] for p in pages),
        n_tied=sum(p["n_tied"] for p in pages), n_det=ND, tp=TP,
        precision=P, recall=R, f1=2 * P * R / (P + R) if P + R else 0,
        recall_aligned_subset=float(np.average(
            [p["recall_aligned"] for p in pages],
            weights=[p["n_gt_aligned"] for p in pages])) if pages else None,
        med_adx=float(np.median(adx)), p90_adx=float(np.percentile(adx, 90)),
        p99_adx=float(np.percentile(adx, 99)),
        med_ady=float(np.median(ady)), p90_ady=float(np.percentile(ady, 90)),
        med_adx_nhw=float(np.median(adx)) / NH_W,
        p90_adx_nhw=float(np.percentile(adx, 90)) / NH_W,
        med_ady_nhh=float(np.median(ady)) / NH_H,
        p90_ady_nhh=float(np.percentile(ady, 90)) / NH_H,
        bias_dx=float(np.mean([r["dx"] for r in mt])),
        bias_dy=float(np.mean([r["dy"] for r in mt])),
        slp_acc=float(np.mean([round(r["det_slp"]) == r["gt_slp"] for r in sl]))
        if sl else None,
        slp_within1=float(np.mean([abs(round(r["det_slp"]) - r["gt_slp"]) <= 1
                                   for r in sl])) if sl else None,
        slp_n=len(sl),
        pitch_upto_accidental_acc=float(np.mean([r["pitch_ok"] for r in pk]))
        if pk else None,
        pitch_n=len(pk),
        sensitivity={"%.1f" % t: dict(
            tol_px=t, tol_nhw=t / NH_W,
            precision=v[0] / v[2] if v[2] else 0,
            recall=v[0] / v[1] if v[1] else 0,
            f1=(2 * v[0] / (v[1] + v[2])) if (v[1] + v[2]) else 0)
            for t, v in sorted(sens.items())},
        wall_per_page_noteheads=float(np.median([p["wall_noteheads"] for p in pages])),
        wall_per_page_total=float(np.median(
            [p["wall_total"] for p in pages if p["wall_total"]])),
        staff_space_px=STAFF_SPACE, notehead_w_px=NH_W, notehead_h_px=NH_H,
    )
    json.dump(dict(aggregate=agg, pages=pages, notes=per_note), open(args.out, "w"))
    if args.quiet:
        return

    print(json.dumps({k: v for k, v in agg.items() if k != "sensitivity"}, indent=1))
    print("\ntolerance sensitivity (micro, all pages):")
    print(" %-8s %-8s %8s %8s %8s" % ("tol_px", "tol/nhw", "prec", "rec", "F1"))
    for v in agg["sensitivity"].values():
        print(" %-8.2f %-8.2f %8.4f %8.4f %8.4f"
              % (v["tol_px"], v["tol_nhw"], v["precision"], v["recall"], v["f1"]))
    print("\nper-page (sorted by F1):")
    print(" %5s %5s %5s %6s %6s %6s %7s %6s %6s %6s  %s"
          % ("n_gt", "n_det", "tp", "P", "R", "F1", "R_algn", "mdx", "slp", "pitch",
             "page"))
    for p in sorted(pages, key=lambda p: p["f1"]):
        print(" %5d %5d %5d %6.3f %6.3f %6.3f %7.3f %6.2f %6s %6s  %s"
              % (p["n_gt"], p["n_det"], p["tp"], p["precision"], p["recall"],
                 p["f1"], p["recall_aligned"],
                 p["med_adx"] if p["med_adx"] is not None else -1,
                 ("%.3f" % p["slp_acc"]) if p["slp_acc"] is not None else "-",
                 ("%.3f" % p["pitch_acc"]) if p["pitch_acc"] is not None else "-",
                 p["key"][:48]))


if __name__ == "__main__":
    main()
