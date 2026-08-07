#!/usr/bin/env python
"""Failure taxonomy + annotated crops for the oemer benchmark.

Reads the JSON written by omr_score.py and answers "where does oemer lose
noteheads": broken down by notehead class, tie, chord size, ledger lines,
local density, staff, and horizontal position on the page.  Each bucket is
reported as a miss rate against the page-set base rate, so a bucket only
counts as a failure mode if it is worse than average.

Also renders annotated crops of the worst page (green = matched,
red = missed, orange = spurious detection).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def bucket_table(notes, name, keyfn, base):
    rows = defaultdict(lambda: [0, 0])
    for r in notes:
        k = keyfn(r)
        if k is None:
            continue
        rows[k][0] += 1
        rows[k][1] += 0 if r["matched"] else 1
    print("\n  %s" % name)
    print("    %-22s %7s %7s %8s %8s" % ("bucket", "n", "missed", "miss%", "vs base"))
    for k in sorted(rows, key=lambda k: (str(type(k)), k)):
        n, mi = rows[k]
        if n < 5:
            continue
        mr = mi / n
        print("    %-22s %7d %7d %7.2f%% %8.2fx"
              % (str(k), n, mi, 100 * mr, mr / base if base else 0))


def draw(work, variant, key, pagejson, out_png, box=None):
    from PIL import Image, ImageDraw
    man = {m["key"]: m for m in json.load(open(work + "/pages/manifest.json"))}
    m = man[key]
    S = m["hires_scale"] if variant == "hires" else 1
    im = Image.open(m["src_img"]).convert("RGB")
    d = ImageDraw.Draw(im)
    notes = [r for r in pagejson["notes"] if r["key"] == key]
    rec = json.load(open("%s/out/%s/%s.json" % (work, variant, key)))
    det = [(n["cx"] / S, n["cy"] / S) for n in rec["notes"]]
    matched_det = set()
    for r in notes:
        x, y = r["gt_x"], r["gt_y"]
        if r["matched"]:
            d.ellipse([x - 7, y - 6, x + 7, y + 6], outline=(0, 160, 0))
            best = min(range(len(det)), key=lambda i: (det[i][0] - x) ** 2
                       + (det[i][1] - y) ** 2)
            matched_det.add(best)
        else:
            d.ellipse([x - 8, y - 7, x + 8, y + 7], outline=(220, 0, 0), width=2)
    for i, (x, y) in enumerate(det):
        if i not in matched_det:
            d.line([x - 6, y - 6, x + 6, y + 6], fill=(255, 140, 0), width=2)
            d.line([x - 6, y + 6, x + 6, y - 6], fill=(255, 140, 0), width=2)
    if box:
        im = im.crop(box)
        im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
    im.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--work", default="/scratch/pmohseni/omr")
    ap.add_argument("--variant", default="native")
    ap.add_argument("--crops", default=None, help="dir to write annotated crops")
    args = ap.parse_args()

    J = json.load(open(args.scored))
    A, pages, notes = J["aggregate"], J["pages"], J["notes"]
    base = 1 - A["recall"]

    print("=" * 78)
    print("oemer on MSMD LilyPond piano pages -- variant=%s  tol=%.1fpx (%.2f notehead widths)"
          % (A["variant"], A["tol_px"], A["tol_px"] / A["notehead_w_px"]))
    print("=" * 78)
    print(" pages %d   GT noteheads %d (all) / %d (MSMD-aligned subset)   detections %d"
          % (A["n_pages"], A["n_gt_all"], A["n_gt_aligned"], A["n_det"]))
    print(" precision %.4f   recall %.4f   F1 %.4f" % (A["precision"], A["recall"], A["f1"]))
    print(" recall on MSMD-aligned subset %.4f" % A["recall_aligned_subset"])
    print(" |dx|  median %.2f px (%.3f nhw)   p90 %.2f px   p99 %.2f px   bias %+.2f"
          % (A["med_adx"], A["med_adx_nhw"], A["p90_adx"], A["p99_adx"], A["bias_dx"]))
    print(" |dy|  median %.2f px (%.3f nhh)   p90 %.2f px                 bias %+.2f"
          % (A["med_ady"], A["med_ady_nhh"], A["p90_ady"], A["bias_dy"]))
    print(" staff-line position  exact %.4f   within +-1 %.4f   (n=%d)"
          % (A["slp_acc"] or 0, A["slp_within1"] or 0, A["slp_n"]))
    if A["pitch_upto_accidental_acc"] is not None:
        print(" pitch up to accidental (staff pos + clef) %.4f  (n=%d)"
              % (A["pitch_upto_accidental_acc"], A["pitch_n"]))
    print(" wall clock  %.1f s/page to noteheads   %.1f s/page full pipeline"
          % (A["wall_per_page_noteheads"], A["wall_per_page_total"]))

    print("\n-- tolerance sensitivity ------------------------------------------------")
    print(" %-8s %-9s %9s %9s %9s" % ("tol_px", "tol/nhw", "precision", "recall", "F1"))
    for v in A["sensitivity"].values():
        print(" %-8.1f %-9.2f %9.4f %9.4f %9.4f"
              % (v["tol_px"], v["tol_nhw"], v["precision"], v["recall"], v["f1"]))

    print("\n-- per page (sorted by F1) ----------------------------------------------")
    print(" %5s %5s %6s %6s %6s %7s %6s %6s %6s %6s  %s"
          % ("n_gt", "n_det", "P", "R", "F1", "R_algn", "mdx", "p90dx", "slp", "pitch", "page"))
    for p in sorted(pages, key=lambda p: p["f1"]):
        print(" %5d %5d %6.3f %6.3f %6.3f %7.3f %6.2f %6.2f %6s %6s  %s"
              % (p["n_gt"], p["n_det"], p["precision"], p["recall"], p["f1"],
                 p["recall_aligned"], p["med_adx"] or -1, p["p90_adx"] or -1,
                 ("%.3f" % p["slp_acc"]) if p["slp_acc"] is not None else "-",
                 ("%.3f" % p["pitch_acc"]) if p["pitch_acc"] is not None else "-",
                 p["key"][:44]))
    f1s = np.array([p["f1"] for p in pages])
    print(" spread: F1 min %.3f  p10 %.3f  median %.3f  max %.3f"
          % (f1s.min(), np.percentile(f1s, 10), np.median(f1s), f1s.max()))

    print("\n-- failure taxonomy (base miss rate %.2f%%) -------------------------------"
          % (100 * base))
    bucket_table(notes, "notehead class", lambda r: r["cls"], base)
    bucket_table(notes, "tie continuation", lambda r: "tied" if r["tied"] else "not tied", base)
    bucket_table(notes, "chord size (noteheads sharing an x column)",
                 lambda r: "%d" % min(r["chord"], 4) + ("+" if r["chord"] >= 4 else ""), base)
    bucket_table(notes, "ledger lines (staff position outside 0..10)",
                 lambda r: ("in staff" if not r["ledger"]
                            else "ledger %d" % min(abs(r["gt_slp"] - 5) // 3, 4))
                 if r["gt_slp"] is not None else None, base)
    bucket_table(notes, "local density (noteheads within 3 staff spaces)",
                 lambda r: "%d" % min(r["dens"], 5) + ("+" if r["dens"] >= 5 else ""), base)
    bucket_table(notes, "clef of host staff", lambda r: r["gt_clef"], base)
    bucket_table(notes, "horizontal position on page (x/835)",
                 lambda r: "%.1f-%.1f" % (np.floor(r["gt_x"] / 835 * 5) / 5,
                                          np.floor(r["gt_x"] / 835 * 5) / 5 + 0.2), base)

    fp = A["n_det"] - A["tp"]
    inval = sum(p["n_invalid"] for p in pages)
    print("\n-- spurious detections ---------------------------------------------------")
    print("  %d detections unmatched at tol=%.1f (%.2f%% of all detections)"
          % (fp, A["tol_px"], 100 * fp / A["n_det"] if A["n_det"] else 0))
    print("  %d detections carry oemer's own `invalid` flag (%.2f%%)"
          % (inval, 100 * inval / A["n_det"] if A["n_det"] else 0))

    if args.crops:
        os.makedirs(args.crops, exist_ok=True)
        worst = sorted(pages, key=lambda p: p["f1"])[0]
        miss = [r for r in notes if r["key"] == worst["key"] and not r["matched"]]
        print("\n-- annotated crops (worst page: %s, F1 %.3f, %d missed) ---"
              % (worst["key"], worst["f1"], len(miss)))
        p = draw(args.work, args.variant, worst["key"], J,
                 os.path.join(args.crops, "worst_full.png"))
        print("  %s" % p)
        if miss:
            ys = np.array([r["gt_y"] for r in miss])
            c = int(np.median(ys))
            p = draw(args.work, args.variant, worst["key"], J,
                     os.path.join(args.crops, "worst_crop.png"),
                     box=(0, max(0, c - 60), 835, c + 60))
            print("  %s" % p)
            print("  worst-page missed noteheads (x, y, slp, pitch, chord, ledger, tied):")
            for r in sorted(miss, key=lambda r: (r["gt_y"], r["gt_x"]))[:40]:
                print("    x=%6.1f y=%6.1f slp=%4s pitch=%3d chord=%d ledger=%d tied=%d %s"
                      % (r["gt_x"], r["gt_y"], r["gt_slp"], r["pitch"], r["chord"],
                         r["ledger"], r["tied"], r["cls"]))


if __name__ == "__main__":
    main()
