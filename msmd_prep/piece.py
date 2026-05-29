"""End-to-end per-piece extraction: strip + MIDI + annotation files."""
from __future__ import annotations
import glob, os, shutil

from .annotate import build_annotations
from .mung import list_performances, parse_noteheads
from .strip import build_strip


def _pick_engraving(piece_dir: str) -> str:
    """Return the score engraving id (the only one in MSMD's no-audio archive)."""
    scores_dir = os.path.join(piece_dir, "scores")
    candidates = sorted(d for d in os.listdir(scores_dir)
                        if os.path.isdir(os.path.join(scores_dir, d)))
    if not candidates:
        raise FileNotFoundError(f"no engraving under {scores_dir}")
    return candidates[0]


def _pick_performance(mung_dir: str, prefer: list[str]) -> str:
    perfs = list_performances(mung_dir)
    if not perfs:
        raise ValueError(f"no performances declared in {mung_dir}")
    for token in prefer:
        for p in perfs:
            if token in p:
                return p
    return perfs[0]


def build_piece(
    piece_dir: str,
    out_dir: str,
    *,
    max_pad_px: int = 40,
    target_raw_h: int = 107,
    performance_prefer: tuple[str, ...] = ("tempo-1000_grand-piano-YDP", "grand-piano-YDP"),
):
    """Produce strip.png + score.midi + annotations.json + noteheads.npz.

    Audio synthesis is NOT performed here (see msmd_prep.synth). The audio.wav
    slot is reserved in annotations.json and the manifest entry.
    """
    os.makedirs(out_dir, exist_ok=True)
    piece_id = os.path.basename(piece_dir.rstrip("/"))

    engraving_id = _pick_engraving(piece_dir)
    score_dir    = os.path.join(piece_dir, "scores", engraving_id)
    mung_dir     = os.path.join(score_dir, "mung")
    performance_id = _pick_performance(mung_dir, prefer=list(performance_prefer))

    # ---- strip ----
    strip, mapping = build_strip(score_dir, max_pad_px=max_pad_px, target_raw_h=target_raw_h)
    strip_path = os.path.join(out_dir, "strip.png")
    strip.save(strip_path)

    # ---- midi ----
    midi_src = os.path.join(piece_dir, "performances", performance_id, f"{performance_id}.midi")
    midi_dst = os.path.join(out_dir, "score.midi")
    shutil.copyfile(midi_src, midi_dst)

    # ---- noteheads ----
    noteheads = list(parse_noteheads(mung_dir, performance_id))

    # ---- annotations ----
    paths = build_annotations(
        piece_id=piece_id,
        score_engraving_id=engraving_id,
        performance_id=performance_id,
        strip_path=strip_path,
        strip_size=strip.size,
        audio_path=None,
        audio_sample_rate=24000,
        audio_soundfont=None,
        audio_peak_db=-3.0,
        midi_path=midi_dst,
        mapping=mapping,
        noteheads=noteheads,
        out_dir=out_dir,
    )
    return {
        "piece_id":           piece_id,
        "score_engraving_id": engraving_id,
        "performance_id":     performance_id,
        "strip":              strip_path,
        "midi":               midi_dst,
        **paths,
        "notehead_count":     len(noteheads),
        "strip_size":         strip.size,
    }
