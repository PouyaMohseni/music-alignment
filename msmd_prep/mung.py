"""Minimal MuNG (MUSCIMA++) XML parser for MSMD notehead annotations."""
from __future__ import annotations
import glob, os
import xml.etree.ElementTree as ET

NOTEHEAD_CLASSES = {"notehead-full", "notehead-empty"}


def _data_items(obj: ET.Element) -> dict:
    out = {}
    for d in obj.findall("Data/DataItem"):
        t = d.attrib.get("type", "str")
        raw = d.text or ""
        if t == "int":
            out[d.attrib["key"]] = int(raw)
        elif t == "float":
            out[d.attrib["key"]] = float(raw)
        else:
            out[d.attrib["key"]] = raw
    return out


def list_performances(mung_dir: str) -> list[str]:
    """Discover performance names from the first XML file's data keys."""
    first = sorted(glob.glob(os.path.join(mung_dir, "*.xml")))[0]
    root = ET.parse(first).getroot()
    names = set()
    for obj in root.iter("CropObject"):
        if obj.findtext("ClassName") not in NOTEHEAD_CLASSES:
            continue
        for d in obj.findall("Data/DataItem"):
            key = d.attrib["key"]
            if key.endswith("_onset_seconds"):
                names.add(key[: -len("_onset_seconds")])
        if names:
            break
    return sorted(names)


def parse_noteheads(mung_dir: str, performance_name: str):
    """Yield one dict per notehead across all pages, in (page, top, left) order.

    Each dict has:
        page_idx, page_x, page_y, midi_pitch, tied,
        onset_sec, midi_offset_sec
    """
    onset_key = f"{performance_name}_onset_seconds"
    dur_key   = f"{performance_name}_duration_seconds"
    for f in sorted(glob.glob(os.path.join(mung_dir, "*.xml"))):
        page_idx = int(os.path.splitext(os.path.basename(f))[0]) - 1
        root = ET.parse(f).getroot()
        objs = [o for o in root.iter("CropObject")
                if o.findtext("ClassName") in NOTEHEAD_CLASSES]
        objs.sort(key=lambda o: (int(o.findtext("Top")), int(o.findtext("Left"))))
        for obj in objs:
            top    = int(obj.findtext("Top"))
            left   = int(obj.findtext("Left"))
            width  = int(obj.findtext("Width"))
            height = int(obj.findtext("Height"))
            data = _data_items(obj)
            if onset_key not in data:
                continue
            onset = data[onset_key]
            dur   = data.get(dur_key, 0.0)
            yield {
                "page_idx":        page_idx,
                "page_x":          left + width // 2,
                "page_y":          top + height // 2,
                "midi_pitch":      data.get("midi_pitch_code", -1),
                "tied":            data.get("tied", 0),
                "onset_sec":       onset,
                "midi_offset_sec": onset + dur,
            }
