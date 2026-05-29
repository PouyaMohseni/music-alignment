"""Annotation schema constants — single source of truth for stages and loaders."""
from __future__ import annotations

SCHEMA_VERSION = "1.0"

# Columnar arrays in noteheads.npz. dtype is what gets stored on disk.
NPZ_DTYPES = {
    "onset_sec":       "float32",
    "midi_offset_sec": "float32",
    "strip_x":         "int32",
    "midi_pitch":      "int8",
    "system_idx":      "int16",
    "measure_idx":     "int16",
    "page_idx":        "int16",
    "page_x":          "int16",
    "page_y":          "int16",
}
NPZ_KEYS = list(NPZ_DTYPES)

# Top-level keys of annotations.json
JSON_KEYS = [
    "schema_version",
    "piece_id",
    "score_engraving_id",
    "performance_id",
    "image",
    "audio",
    "tempo_events",
    "beat_times_sec",
    "bar_times_sec",
    "strip_to_page_mapping",
    "notehead_count",
]

PER_PIECE_FILES = {
    "image":       "strip.png",
    "audio":       "audio.wav",       # written by Stage 2 (FluidSynth)
    "midi":        "score.midi",      # source MIDI of the chosen performance
    "annotations": "annotations.json",
    "noteheads":   "noteheads.npz",
}
