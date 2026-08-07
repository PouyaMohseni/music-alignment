#!/usr/bin/env python
"""Run oemer on one score page and dump every detected notehead with its
coordinates mapped back into the ORIGINAL page pixel space.

Why this exists instead of just calling `oemer <img>`:

  * oemer's `inference.resize_image` rescales every input to a fixed pixel
    budget (3.0M-4.35M px).  All of oemer's layers -- including
    `NoteHead.bbox` -- therefore live in that INTERNAL resized space, not in
    the coordinates of the file you handed it.  The CLI never tells you the
    ratio.  We record it and map back.
  * The CLI's last stages (rhythm parsing + MusicXML building) are the most
    failure-prone, and a crash there would cost us the notehead measurement
    that we actually care about.  Here the notehead stages are committed
    before the fragile stages are attempted, inside try/except.

Output: one JSON per page (see `--out`).
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="page image to run oemer on")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--deskew",
        action="store_true",
        help="run oemer's dewarp stage. Off by default: these pages are "
        "synthetic LilyPond renders with exactly zero skew, and dewarp "
        "applies a non-linear warp that would break the linear "
        "resized->original coordinate mapping.",
    )
    ap.add_argument("--musicxml", default=None, help="also try to build MusicXML here")
    args = ap.parse_args()

    # onnxruntime builds its own thread pool sized to the machine's core count,
    # NOT to the cgroup SLURM gives us: on a 64-core node each worker spawned 64
    # intra-op threads plus their arenas, and N concurrent workers were OOM-killed
    # (rc=137) before finishing a page.  OMP_NUM_THREADS does not control this;
    # only SessionOptions does, and oemer constructs its sessions internally, so
    # patch the constructor.
    import onnxruntime as rt

    _orig_session = rt.InferenceSession

    def _pinned_session(path, *a, **kw):
        so = rt.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("OEMER_THREADS", "4"))
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        kw.setdefault("sess_options", so)
        return _orig_session(path, *a, **kw)

    rt.InferenceSession = _pinned_session

    import cv2
    from oemer import layers
    from oemer.ete import generate_pred, register_note_id
    from oemer.dewarp import estimate_coords, dewarp
    from oemer.staffline_extraction import extract as staff_extract
    from oemer.notehead_extraction import extract as note_extract
    from oemer.note_group_extraction import extract as group_extract

    rec = {"img": args.img, "deskew": bool(args.deskew), "stages": {}, "errors": []}
    orig_w, orig_h = Image.open(args.img).size
    rec["orig_size"] = [orig_w, orig_h]

    t_all = time.time()

    # ---- neural inference (the expensive part) -------------------------
    t = time.time()
    staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(args.img)
    rec["stages"]["generate_pred"] = time.time() - t
    pred_h, pred_w = staff.shape
    rec["pred_size"] = [int(pred_w), int(pred_h)]
    # oemer's internal resize factor. Map back with x_orig = x_pred / sx.
    rec["sx"] = pred_w / orig_w
    rec["sy"] = pred_h / orig_h

    image = cv2.imread(args.img)
    image = cv2.resize(image, (pred_w, pred_h))

    if args.deskew:
        t = time.time()
        cx, cy = estimate_coords(staff)
        staff = dewarp(staff, cx, cy)
        symbols = dewarp(symbols, cx, cy)
        stems_rests = dewarp(stems_rests, cx, cy)
        clefs_keys = dewarp(clefs_keys, cx, cy)
        notehead = dewarp(notehead, cx, cy)
        for i in range(image.shape[2]):
            image[..., i] = dewarp(image[..., i], cx, cy)
        rec["stages"]["dewarp"] = time.time() - t

    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols > 1] = 1
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)

    # ---- stafflines ----------------------------------------------------
    t = time.time()
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)
    layers.register_layer("zones", zones)
    rec["stages"]["staff_extract"] = time.time() - t

    st = []
    for tr in np.ndindex(staffs.shape):
        s = staffs[tr]
        if s is None or not getattr(s, "lines", None):
            continue
        try:
            st.append(
                dict(
                    idx=[int(v) for v in tr],
                    track=None if s.track is None else int(s.track),
                    group=None if s.group is None else int(s.group),
                    y_center=float(s.y_center),
                    y_upper=float(s.y_upper),
                    y_lower=float(s.y_lower),
                    x_left=float(s.x_left),
                    x_right=float(s.x_right),
                    unit_size=float(s.unit_size),
                    is_interp=bool(s.is_interp),
                    line_y=[float(l.y_center) for l in s.lines],
                )
            )
        except Exception:
            rec["errors"].append("staff dump: " + traceback.format_exc())
    rec["staffs"] = st

    # ---- noteheads (the measurement) -----------------------------------
    t = time.time()
    notes = note_extract()
    layers.register_layer("notes", np.array(notes))
    layers.register_layer("note_id", np.zeros(symbols.shape, dtype=np.int64) - 1)
    register_note_id()
    rec["stages"]["note_extract"] = time.time() - t

    t = time.time()
    groups, group_map = group_extract()
    layers.register_layer("note_groups", np.array(groups))
    layers.register_layer("group_map", group_map)
    rec["stages"]["group_extract"] = time.time() - t

    def dump_notes():
        out = []
        for n in notes:
            x1, y1, x2, y2 = [float(v) for v in n.bbox]
            lab = n._label
            out.append(
                dict(
                    id=None if n.id is None else int(n.id),
                    # bbox in oemer's internal resized space
                    bbox_pred=[x1, y1, x2, y2],
                    # centre mapped back into ORIGINAL page pixels
                    cx=((x1 + x2) / 2) / rec["sx"],
                    cy=((y1 + y2) / 2) / rec["sy"],
                    w=(x2 - x1) / rec["sx"],
                    h=(y2 - y1) / rec["sy"],
                    staff_line_pos=None
                    if n.staff_line_pos is None
                    else float(n.staff_line_pos),
                    track=None if n.track is None else int(n.track),
                    group=None if n.group is None else int(n.group),
                    note_group_id=None
                    if n.note_group_id is None
                    else int(n.note_group_id),
                    label=None if lab is None else str(lab),
                    invalid=bool(n.invalid),
                    has_dot=bool(n.has_dot),
                    stem_up=None if n.stem_up is None else bool(n.stem_up),
                    npoints=len(n.points),
                    sfn=None if n.sfn is None else str(n.sfn),
                )
            )
        return out

    rec["notes"] = dump_notes()
    rec["n_notes"] = len(rec["notes"])

    # commit the notehead result before touching the fragile stages
    rec["wall_to_noteheads"] = time.time() - t_all
    with open(args.out, "w") as f:
        json.dump(rec, f)

    # ---- fragile downstream stages: clefs / rhythm / MusicXML ----------
    try:
        from oemer.symbol_extraction import extract as symbol_extract
        from oemer.rhythm_extraction import extract as rhythm_extract
        from oemer.build_system import MusicXMLBuilder

        t = time.time()
        barlines, clefs, sfns, rests = symbol_extract()
        layers.register_layer("barlines", np.array(barlines))
        layers.register_layer("clefs", np.array(clefs))
        layers.register_layer("sfns", np.array(sfns))
        layers.register_layer("rests", np.array(rests))
        rec["stages"]["symbol_extract"] = time.time() - t
        rec["clefs"] = [
            dict(
                label=str(c.label),
                track=None if c.track is None else int(c.track),
                group=None if c.group is None else int(c.group),
                cx=float((c.bbox[0] + c.bbox[2]) / 2) / rec["sx"],
                cy=float((c.bbox[1] + c.bbox[3]) / 2) / rec["sy"],
            )
            for c in clefs
        ]
        rec["n_barlines"] = len(barlines)
        rec["n_rests"] = len(rests)
        rec["n_sfns"] = len(sfns)

        t = time.time()
        rhythm_extract()
        rec["stages"]["rhythm_extract"] = time.time() - t
        # rhythm_extract mutates note labels/sfn in place -> re-dump
        rec["notes"] = dump_notes()

        if args.musicxml:
            t = time.time()
            b = MusicXMLBuilder(title=os.path.basename(args.img))
            b.build()
            with open(args.musicxml, "wb") as f:
                f.write(b.to_musicxml())
            rec["stages"]["musicxml"] = time.time() - t
            rec["musicxml"] = args.musicxml
    except Exception:
        rec["errors"].append(traceback.format_exc())

    rec["wall_total"] = time.time() - t_all
    with open(args.out, "w") as f:
        json.dump(rec, f)
    print(
        "OK %s notes=%d noteheads_wall=%.1fs total=%.1fs errors=%d"
        % (
            os.path.basename(args.img),
            rec["n_notes"],
            rec["wall_to_noteheads"],
            rec["wall_total"],
            len(rec["errors"]),
        )
    )


if __name__ == "__main__":
    sys.exit(main())
