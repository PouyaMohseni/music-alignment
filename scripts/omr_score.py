#!/usr/bin/env python
"""Score oemer's notehead output against MSMD's LilyPond ground truth.

Everything is computed in *native MSMD page pixels* (835x1181, staff space
7.0 px).  For the `hires` variant, oemer's coordinates are divided by the
render scale first, so the two variants' error numbers are directly
comparable.

Notehead scale used for normalising errors:
    staff_space  s  = 7.0 px  (measured, identical on every MSMD page)
    notehead w   ~= 1.3 * s = 9.1 px   (LilyPond black notehead glyph)
    notehead h   ~= 1.0 * s = 7.0 px

Ground-truth staff-line position replicates oemer's own convention exactly
(oemer/notehead_extraction.py:341-367): index 0 == the half-space below the
bottom staff line (D4 read in treble clef), 1 == bottom line, 9 == top line,
increasing upward.  GT staff lines are read straight off the page image
(rows that are >70% black across the system's x-span), so the comparison is
geometric on both sides and does not depend on oemer's own staff detection
being right.
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
NH_W = 1.3 * STAFF_SPACE  # 9.1 px
NH_H = 1.0 * STAFF_SPACE


# ----------------------------------------------------------------- GT staves
def gt_staves(img_path, systems):
    """Recover each staff's 5 line y-positions from the page image itself."""
    blk = np.array(Image.open(img_path).convert("L")) < 128
    staves = []
    for b in systems:
        y0, y1 = int(b[:, 0].min()), int(b[:, 0].max())
        x0, x1 = int(b[:, 1].min()), int(b[:, 1].max())
        prof = blk[y0 : y1 + 1, x0 : x1 + 1].mean(1)
        rows = np.where(prof > 0.7)[0]
        if len(rows) < 5:
            continue
        grp = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
        cy = np.array([g.mean() + y0 for g in grp])
        # split the run of lines into groups of 5 (one grand staff = 10 lines)
        gaps = np.diff(cy)
        med = np.median(gaps[gaps < 3 * STAFF_SPACE]) if len(gaps) else STAFF_SPACE
        cuts = np.where(gaps > 2.5 * med)[0] + 1
        for chunk in np.split(cy, cuts):
            if len(chunk) == 5:
                staves.append(dict(lines=chunk.tolist(), x0=x0, x1=x1))
            elif len(chunk) > 5 and len(chunk) % 5 == 0:
                for k in range(0, len(chunk), 5):
                    staves.append(dict(lines=chunk[k : k + 5].tolist(), x0=x0, x1=x1))
    for s in staves:
        L = np.array(s["lines"])
        s["y_upper"], s["y_lower"] = float(L.min()), float(L.max())
        s["y_center"] = float(L.mean())
        s["bottom"] = float(L.max())
        s["space"] = float(np.median(np.diff(np.sort(L))))
    return staves


def gt_staff_line_pos(y, x, staves):
    """oemer's staff_line_pos convention, computed from GT geometry."""
    if not staves:
        return None, None
    # oemer picks the closest staff by centre, preferring one that contains y
    cand = [s for s in staves if s["y_upper"] - 3 * STAFF_SPACE <= y <= s["y_lower"] + 3 * STAFF_SPACE
            and s["x0"] - 20 <= x <= s["x1"] + 20]
    pool = cand if cand else staves
    i = int(np.argmin([abs(s["y_center"] - y) for s in pool]))
    s = pool[i]
    step = s["space"] / 2.0
    pos = int(round((s["bottom"] - y) / step)) + 1
    return pos, staves.index(s)


# ----------------------------------------------------------------- matching
def match(gt, det, tol):
    """Optimal 1-1 assignment under a Euclidean gate of `tol` px.

    Hungarian rather than greedy: in a dense chord, greedy nearest-first can
    lock a detection onto the wrong notehead and cascade.
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
    ap.add_argument("--drop-invalid", action="store_true",
                    help="discard detections oemer itself flagged invalid")
    args = ap.parse_args()

    man = {m["key"]: m for m in json.load(open(args.work + "/pages/manifest.json"))}
    outdir = os.path.join(args.work, "out", args.variant)

    TOLS = [2.0, 3.0, 4.55, 6.0, 7.0, 9.1, 13.65]  # px; 4.55 = half notehead width
    pages, per_note = [], []
    sens = defaultdict(lambda: [0, 0, 0])  # tol -> [tp, ngt, ndet]

    for f in sorted(glob.glob(outdir + "/*.json")):
        rec = json.load(open(f))
        key = os.path.basename(f)[:-5]
        m = man[key]
        S = m["hires_scale"] if args.variant == "hires" else 1

        g = np.load(m["gt"])
        gtyx = g["notes"]                       # (N,2) [y,x] native page px
        gt = np.stack([gtyx[:, 1], gtyx[:, 0]], 1)  # -> (x,y)
        systems = g["systems"]

        notes = rec["notes"]
        if args.drop_invalid:
            notes = [n for n in notes if not n["invalid"]]
        det = np.array([[n["cx"] / S, n["cy"] / S] for n in notes]).reshape(-1, 2)

        pairs = match(gt, det, args.tol)
        tp = len(pairs)
        prec = tp / len(det) if len(det) else 0.0
        rec_ = tp / len(gt) if len(gt) else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0

        for t in TOLS:
            p2 = match(gt, det, t)
            s = sens[t]
            s[0] += len(p2); s[1] += len(gt); s[2] += len(det)

        stv = gt_staves(m["src_img"], systems)
        dx = det[pairs[:, 1], 0] - gt[pairs[:, 0], 0] if tp else np.array([])
        dy = det[pairs[:, 1], 1] - gt[pairs[:, 0], 1] if tp else np.array([])

        slp_ok = slp_n = 0
        gi2di = {int(a): int(b) for a, b in pairs}
        for gi in range(len(gt)):
            gp, gs = gt_staff_line_pos(gtyx[gi, 0], gtyx[gi, 1], stv)
            # local context, computed from GT alone (independent of oemer)
            near = np.linalg.norm(gt - gt[gi], axis=1)
            chord = int(np.sum((np.abs(gt[:, 0] - gt[gi, 0]) < 0.6 * NH_W)
                               & (np.abs(gt[:, 1] - gt[gi, 1]) < 4 * STAFF_SPACE)) - 1)
            dens = int(np.sum(near < 3 * STAFF_SPACE) - 1)
            ledger = gp is not None and (gp < 0 or gp > 10)
            row = dict(key=key, gi=gi, matched=gi in gi2di,
                       gt_x=float(gt[gi, 0]), gt_y=float(gt[gi, 1]),
                       gt_slp=gp, gt_staff=gs, chord=chord, dens=dens,
                       ledger=bool(ledger))
            if gi in gi2di:
                n = notes[gi2di[gi]]
                row["dx"] = float(det[gi2di[gi], 0] - gt[gi, 0])
                row["dy"] = float(det[gi2di[gi], 1] - gt[gi, 1])
                row["det_slp"] = n["staff_line_pos"]
                row["invalid"] = n["invalid"]
                row["label"] = n["label"]
                if gp is not None and n["staff_line_pos"] is not None:
                    slp_n += 1
                    slp_ok += int(round(n["staff_line_pos"]) == gp)
            per_note.append(row)

        pages.append(dict(
            key=key, piece=m["piece"], page=m["page"],
            n_gt=len(gt), n_det=len(det),
            n_invalid=sum(1 for n in rec["notes"] if n["invalid"]),
            tp=tp, precision=prec, recall=rec_, f1=f1,
            med_adx=float(np.median(np.abs(dx))) if tp else None,
            p90_adx=float(np.percentile(np.abs(dx), 90)) if tp else None,
            med_ady=float(np.median(np.abs(dy))) if tp else None,
            p90_ady=float(np.percentile(np.abs(dy), 90)) if tp else None,
            bias_dx=float(np.mean(dx)) if tp else None,
            bias_dy=float(np.mean(dy)) if tp else None,
            slp_acc=(slp_ok / slp_n) if slp_n else None, slp_n=slp_n,
            n_gt_staves=len(stv), n_det_staves=len(rec.get("staffs", [])),
            wall_noteheads=rec.get("wall_to_noteheads"),
            wall_total=rec.get("wall_total"),
            stages=rec.get("stages"),
            n_errors=len(rec.get("errors", [])),
        ))

    # ---------------- aggregate ----------------
    TP = sum(p["tp"] for p in pages); NG = sum(p["n_gt"] for p in pages)
    ND = sum(p["n_det"] for p in pages)
    P = TP / ND if ND else 0; R = TP / NG if NG else 0
    adx = np.abs([r["dx"] for r in per_note if r["matched"]])
    ady = np.abs([r["dy"] for r in per_note if r["matched"]])
    sl = [r for r in per_note if r["matched"] and r.get("det_slp") is not None
          and r["gt_slp"] is not None]
    agg = dict(
        variant=args.variant, tol_px=args.tol, drop_invalid=args.drop_invalid,
        n_pages=len(pages), n_gt=NG, n_det=ND, tp=TP,
        precision=P, recall=R, f1=2 * P * R / (P + R) if P + R else 0,
        med_adx=float(np.median(adx)), p90_adx=float(np.percentile(adx, 90)),
        med_ady=float(np.median(ady)), p90_ady=float(np.percentile(ady, 90)),
        med_adx_nhw=float(np.median(adx)) / NH_W,
        p90_adx_nhw=float(np.percentile(adx, 90)) / NH_W,
        med_ady_nhh=float(np.median(ady)) / NH_H,
        p90_ady_nhh=float(np.percentile(ady, 90)) / NH_H,
        slp_acc=float(np.mean([round(r["det_slp"]) == r["gt_slp"] for r in sl])) if sl else None,
        slp_within1=float(np.mean([abs(round(r["det_slp"]) - r["gt_slp"]) <= 1 for r in sl])) if sl else None,
        slp_n=len(sl),
        sensitivity={str(t): dict(tol_px=t, tol_nhw=t / NH_W,
                                  precision=v[0] / v[2] if v[2] else 0,
                                  recall=v[0] / v[1] if v[1] else 0,
                                  f1=(2 * v[0] / (v[1] + v[2])) if (v[1] + v[2]) else 0)
                     for t, v in sorted(sens.items())},
        wall_per_page_noteheads=float(np.median([p["wall_noteheads"] for p in pages])),
        wall_per_page_total=float(np.median([p["wall_total"] for p in pages
                                             if p["wall_total"]])),
        staff_space_px=STAFF_SPACE, notehead_w_px=NH_W,
    )
    json.dump(dict(aggregate=agg, pages=pages, notes=per_note),
              open(args.out, "w"))

    print(json.dumps({k: v for k, v in agg.items() if k != "sensitivity"}, indent=1))
    print("\ntolerance sensitivity (micro over all pages):")
    print(" %-9s %-9s %8s %8s %8s" % ("tol_px", "tol/nhw", "prec", "rec", "F1"))
    for t, v in agg["sensitivity"].items():
        print(" %-9.2f %-9.2f %8.4f %8.4f %8.4f"
              % (v["tol_px"], v["tol_nhw"], v["precision"], v["recall"], v["f1"]))
    print("\nper-page:")
    print(" %5s %5s %5s %6s %6s %6s %6s %6s %6s  %s"
          % ("n_gt", "n_det", "tp", "P", "R", "F1", "mdx", "mdy", "slp", "page"))
    for p in sorted(pages, key=lambda p: p["f1"]):
        print(" %5d %5d %5d %6.3f %6.3f %6.3f %6.2f %6.2f %6s  %s"
              % (p["n_gt"], p["n_det"], p["tp"], p["precision"], p["recall"],
                 p["f1"], p["med_adx"] or -1, p["med_ady"] or -1,
                 ("%.3f" % p["slp_acc"]) if p["slp_acc"] is not None else "-",
                 p["key"][:52]))


if __name__ == "__main__":
    main()
