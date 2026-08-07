#!/usr/bin/env python
"""Select MSMD *page* images (NOT unrolled strips) + their LilyPond ground truth,
and materialise them into a work dir for the oemer benchmark.

Source of truth
---------------
`data/MSMD/msmd_aug_v1-1_no-audio/<piece>/scores/<piece>_ly/`
    img/NN.png          -- the page render, 835x1181 portrait, systems stacked
                           vertically.  This is the real page, not the
                           horizontally-concatenated strip in cpjku_fmt/.
    coords/notes_NN.npy -- (N, 2) float64, [y, x], in the page PNG's OWN pixel
                           space.  Verified: 100% of the points land on black
                           pixels.  No scale_factor is applied here (the /3 in
                           eval_any_cpu.sh belongs to the strip pipeline).
    coords/systems_NN.npy -- (S, 4, 2) system bounding quads, [y, x] corners.

Two image variants are emitted:
    native : the MSMD PNG as stored (835x1181, staff space 7.0 px)
    hires  : the same page re-rendered from <piece>_ly.pdf at 3x
             (2506x3543, staff space 21 px), GT coords scaled by 3.
oemer normalises every input to a ~3.7M px working size regardless, so the
pair isolates *render sharpness* from resolution.
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSMD = os.path.join(REPO, "data/MSMD/msmd_aug_v1-1_no-audio")
SPLITS = os.path.join(REPO, "data/MSMD/processed/splits.json")


def candidates(split="test"):
    pieces = json.load(open(SPLITS))[split]
    rows = []
    for p in pieces:
        d = os.path.join(MSMD, p, "scores", p + "_ly")
        if not os.path.isdir(d):
            continue
        pdfs = glob.glob(d + "/*.pdf")
        for im in sorted(glob.glob(d + "/img/*.png")):
            pg = os.path.basename(im)[:-4]
            nf = os.path.join(d, "coords", "notes_%s.npy" % pg)
            sf = os.path.join(d, "coords", "systems_%s.npy" % pg)
            if not os.path.exists(nf):
                continue
            rows.append(
                dict(
                    piece=p,
                    page=pg,
                    img=im,
                    notes=nf,
                    systems=sf if os.path.exists(sf) else None,
                    pdf=pdfs[0] if pdfs else None,
                    nnotes=int(len(np.load(nf))),
                    nsys=int(len(np.load(sf))) if os.path.exists(sf) else 0,
                )
            )
    return rows


def select(rows, n):
    """One page per piece, stratified evenly over notehead density.

    Prefer a non-first page where the piece has one, so the sample is not
    entirely page 1 (title block, indented first system).
    """
    byp = {}
    for r in rows:
        byp.setdefault(r["piece"], []).append(r)
    one = []
    for v in byp.values():
        v = sorted(v, key=lambda r: r["page"])
        one.append(v[1] if len(v) > 1 else v[0])
    one.sort(key=lambda r: (r["nnotes"], r["piece"]))
    idx = np.linspace(0, len(one) - 1, n).round().astype(int)
    return [one[i] for i in sorted(set(idx.tolist()))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--hires-scale", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for sub in ("native", "hires", "gt"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    sel = select(candidates(args.split), args.n)
    man = []
    for r in sel:
        key = "%s__p%s" % (r["piece"], r["page"])
        gt = np.load(r["notes"])  # (N,2) [y,x] in page pixels
        sysb = np.load(r["systems"]) if r["systems"] else np.zeros((0, 4, 2))

        # native
        im = Image.open(r["img"])
        w, h = im.size
        nat = os.path.join(args.out, "native", key + ".png")
        im.convert("RGB").save(nat)

        # hires: re-render the same page from the LilyPond PDF
        hi = None
        S = args.hires_scale
        if r["pdf"]:
            pno = int(r["page"])
            pre = os.path.join(args.out, "hires", key)
            subprocess.run(
                [
                    "pdftoppm", "-png", "-f", str(pno), "-l", str(pno),
                    "-scale-to-x", str(w * S), "-scale-to-y", str(h * S),
                    r["pdf"], pre,
                ],
                check=True,
            )
            got = glob.glob(pre + "-*.png")
            if got:
                hi = pre + ".png"
                os.replace(got[0], hi)
                hw, hh = Image.open(hi).size
                assert (hw, hh) == (w * S, h * S), (hw, hh, w * S, h * S)

        np.savez(
            os.path.join(args.out, "gt", key + ".npz"),
            notes=gt.astype(np.float64),
            systems=sysb.astype(np.float64),
        )
        man.append(
            dict(
                key=key, piece=r["piece"], page=r["page"],
                src_img=r["img"], src_pdf=r["pdf"],
                native=nat, hires=hi, hires_scale=S,
                gt=os.path.join(args.out, "gt", key + ".npz"),
                orig_size=[w, h], n_gt=int(len(gt)), n_systems=int(len(sysb)),
            )
        )
        print("%-70s %4d notes  %d sys" % (key, len(gt), len(sysb)))

    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print("\n%d pages, %d GT noteheads -> %s"
          % (len(man), sum(m["n_gt"] for m in man), args.out))


if __name__ == "__main__":
    main()
