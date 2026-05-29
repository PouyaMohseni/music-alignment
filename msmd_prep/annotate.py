"""Stage 5 — given strip mapping + MuNG-parsed noteheads, produce annotation files."""
from __future__ import annotations
import hashlib, json, os, struct, wave
from collections import defaultdict
import numpy as np

from .schema import NPZ_DTYPES, NPZ_KEYS, SCHEMA_VERSION


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_midi_timing(midi_path: str):
    """Return tempo_events, beat_times_sec, bar_times_sec, total_duration_sec.

    Uses pretty_midi if available; otherwise raises ImportError.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    tempo_change_times, tempi = pm.get_tempo_changes()
    tempo_events = [
        {"time_sec": float(t), "bpm": float(bpm)}
        for t, bpm in zip(tempo_change_times, tempi)
    ]
    beat_times = pm.get_beats().tolist()
    bar_times  = pm.get_downbeats().tolist()
    duration   = float(pm.get_end_time())
    return tempo_events, beat_times, bar_times, duration


def _measure_idx(onset_sec: np.ndarray, bar_times_sec: list[float]) -> np.ndarray:
    if not bar_times_sec:
        return np.zeros_like(onset_sec, dtype=np.int16) - 1
    return (np.searchsorted(np.asarray(bar_times_sec), onset_sec, side="right") - 1
            ).astype(np.int16)


def _strip_x_for(page_idx: int, page_x: int, page_y: int,
                 mapping: list[dict]) -> tuple[int, int]:
    """Route the notehead to a system by (page_x, page_y) containment and
    translate to strip coordinates. Uses each system's *expanded* bbox to
    route (so ledger-line noteheads outside the raw staff bbox still get the
    right system), then translates strip_x off the raw bbox's left edge."""
    candidates = [(i, m) for i, m in enumerate(mapping) if m["page_idx"] == page_idx]
    if not candidates:
        raise ValueError(f"no systems on page {page_idx}")
    # 1) exact expanded-bbox containment
    for i, m in candidates:
        x0, y0, x1, y1 = m["exp_bbox"]
        if x0 <= page_x <= x1 and y0 <= page_y <= y1:
            return m["strip_x_start"] + (page_x - m["raw_bbox"][0]), i
    # 2) nearest system by vertical distance to the raw staff bbox
    def vdist(m):
        y0, y1 = m["raw_bbox"][1], m["raw_bbox"][3]
        if y0 <= page_y <= y1:
            return 0
        return min(abs(page_y - y0), abs(page_y - y1))
    i, m = min(candidates, key=lambda im: vdist(im[1]))
    x_min, x_max = m["raw_bbox"][0], m["raw_bbox"][2]
    page_x_clamped = max(x_min, min(x_max, page_x))
    return m["strip_x_start"] + (page_x_clamped - x_min), i


def build_annotations(
    *,
    piece_id: str,
    score_engraving_id: str,
    performance_id: str,
    strip_path: str,
    strip_size: tuple[int, int],         # (width, height)
    audio_path: str | None,              # may not exist yet
    audio_sample_rate: int | None,
    audio_soundfont: str | None,
    audio_peak_db: float | None,
    midi_path: str,
    mapping: list[dict],
    noteheads: list[dict],
    out_dir: str,
):
    """Write annotations.json + noteheads.npz to out_dir."""
    # ---- timing from MIDI ----
    tempo_events, beat_times, bar_times, duration = _read_midi_timing(midi_path)

    # ---- columnar arrays ----
    by_col = defaultdict(list)
    for n in noteheads:
        sx, sys_idx = _strip_x_for(n["page_idx"], n["page_x"], n["page_y"], mapping)
        by_col["onset_sec"].append(n["onset_sec"])
        by_col["midi_offset_sec"].append(n["midi_offset_sec"])
        by_col["strip_x"].append(sx)
        by_col["midi_pitch"].append(n["midi_pitch"])
        by_col["system_idx"].append(sys_idx)
        by_col["page_idx"].append(n["page_idx"])
        by_col["page_x"].append(n["page_x"])
        by_col["page_y"].append(n["page_y"])

    arrs = {k: np.asarray(by_col[k]) for k in NPZ_KEYS if k != "measure_idx"}
    order = np.argsort(arrs["onset_sec"], kind="stable")
    for k in arrs:
        arrs[k] = arrs[k][order]
    arrs["measure_idx"] = _measure_idx(arrs["onset_sec"], bar_times)
    arrs = {k: arrs[k].astype(NPZ_DTYPES[k]) for k in NPZ_KEYS}

    npz_path = os.path.join(out_dir, "noteheads.npz")
    np.savez(npz_path, **arrs)

    # ---- JSON sidecar ----
    image = {
        "path":           "strip.png",
        "sha256":         _sha256(strip_path),
        "width_px":       int(strip_size[0]),
        "height_px":      int(strip_size[1]),
    }
    audio = {
        "path":           "audio.wav",
        "sha256":         _sha256(audio_path) if audio_path and os.path.exists(audio_path) else None,
        "sample_rate_hz": audio_sample_rate,
        "duration_sec":   duration,
        "soundfont":      audio_soundfont,
        "peak_db":        audio_peak_db,
        "synthesized":    bool(audio_path and os.path.exists(audio_path)),
    }
    payload = {
        "schema_version":        SCHEMA_VERSION,
        "piece_id":              piece_id,
        "score_engraving_id":    score_engraving_id,
        "performance_id":        performance_id,
        "image":                 image,
        "audio":                 audio,
        "midi":                  {"path": "score.midi", "sha256": _sha256(midi_path)},
        "tempo_events":          tempo_events,
        "beat_times_sec":        beat_times,
        "bar_times_sec":         bar_times,
        "strip_to_page_mapping": mapping,
        "notehead_count":        int(len(arrs["onset_sec"])),
    }
    json_path = os.path.join(out_dir, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return {"annotations": json_path, "noteheads": npz_path}
