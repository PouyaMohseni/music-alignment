#!/usr/bin/env python
"""Build the stronger ground truth for the oemer benchmark from MSMD's MUNG graphs.

`coords/notes_NN.npy` is the *aligned onset* subset: it omits every notehead
MSMD could not tie to a MIDI onset -- above all the second half of a tie,
which is still ink on the page that any OMR system will (correctly) detect.
Scoring detection against it alone would charge oemer a false positive for
being right.  `mung/NN.xml` is the full notation graph and carries, per
notehead:

    Top/Left/Width/Height  exact notehead bbox in page pixels
    midi_pitch_code        exact MIDI pitch
    tied                   1 if this is the continuation of a tie

Measured on MSMD pages: notehead bbox is 10 x 8 px (staff space 7.0), and
MUNG bbox centres agree with `notes_NN.npy` coordinates to a median of
0.7 px -- which is the noise floor of any coordinate error we report.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np


def parse_mung(path):
    root = ET.parse(path).getroot()
    out = []
    for o in root.findall(".//CropObject"):
        cls = o.find("ClassName").text
        if not cls.startswith("notehead"):
            continue
        d = {i.get("key"): i.text for i in o.findall("./Data/DataItem")}
        t = int(o.find("Top").text)
        l = int(o.find("Left").text)
        w = int(o.find("Width").text)
        h = int(o.find("Height").text)
        out.append(
            dict(
                cls=cls, x=l + w / 2.0, y=t + h / 2.0, w=w, h=h,
                pitch=int(d["midi_pitch_code"]) if d.get("midi_pitch_code") else -1,
                tied=int(d.get("tied", 0) or 0),
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/scratch/pmohseni/omr")
    args = ap.parse_args()
    man = json.load(open(args.work + "/pages/manifest.json"))
    tot = tied = 0
    for m in man:
        d = os.path.dirname(os.path.dirname(m["src_img"]))
        f = os.path.join(d, "mung", "%s.xml" % m["page"])
        nh = parse_mung(f)
        arr = np.array([[n["x"], n["y"], n["w"], n["h"], n["pitch"], n["tied"]]
                        for n in nh], dtype=np.float64).reshape(-1, 6)
        cls = np.array([n["cls"] for n in nh])
        # nearest-neighbour distance from each MSMD-aligned coord to a MUNG
        # notehead: the ground truth's own agreement with itself.
        g = np.load(m["gt"])["notes"]  # (N,2) [y,x]
        if len(g) and len(arr):
            dd = np.linalg.norm(
                np.stack([g[:, 1], g[:, 0]], 1)[:, None, :] - arr[None, :, :2], axis=2
            ).min(1)
        else:
            dd = np.zeros(0)
        np.savez(os.path.join(args.work, "pages", "gt", m["key"] + "_mung.npz"),
                 nh=arr, cls=cls)
        tot += len(arr); tied += int(arr[:, 5].sum())
        print("%-52s mung=%4d aligned=%4d tied=%3d  w=%.0f h=%.0f  gtnoise med=%.2f max=%.2f"
              % (m["key"][:52], len(arr), len(g), int(arr[:, 5].sum()),
                 np.median(arr[:, 2]), np.median(arr[:, 3]),
                 np.median(dd) if len(dd) else -1, dd.max() if len(dd) else -1))
    print("\ntotal MUNG noteheads %d (tied %d)" % (tot, tied))


if __name__ == "__main__":
    main()
