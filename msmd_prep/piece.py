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


def _build_one_performance(
    piece_dir: str,
    out_dir: str,
    piece_id: str,
    engraving_id: str,
    score_dir: str,
    mung_dir: str,
    strip,
    mapping: list,
    strip_path: str,
    performance_id: str,
) -> dict:
    """Build annotations for one performance. Strip is pre-rendered and shared."""
    os.makedirs(out_dir, exist_ok=True)

    midi_src = os.path.join(piece_dir, "performances", performance_id, f"{performance_id}.midi")
    if not os.path.exists(midi_src):
        raise FileNotFoundError(f"MIDI not found: {midi_src}")
    midi_dst = os.path.join(out_dir, "score.midi")
    shutil.copyfile(midi_src, midi_dst)

    # Strip is shared across all performances of the same piece — symlink it
    out_strip = os.path.join(out_dir, "strip.png")
    if not os.path.exists(out_strip):
        os.symlink(os.path.abspath(strip_path), out_strip)

    noteheads = list(parse_noteheads(mung_dir, performance_id))
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


def build_piece(
    piece_dir: str,
    out_dir: str,
    *,
    max_pad_px: int = 40,
    target_raw_h: int = 107,
    performance_prefer: tuple[str, ...] = ("tempo-1000_grand-piano-YDP", "grand-piano-YDP"),
    all_performances: bool = False,
) -> list[dict]:
    """Produce strip.png + per-performance {score.midi, annotations.json, noteheads.npz}.

    With all_performances=False (default): one canonical performance, output in out_dir.
    With all_performances=True: all performances, each in out_dir/<performance_id>/.
    Strip is rendered once and shared (symlinked) across all performances.

    Returns a list of result dicts (one per performance).
    """
    piece_id = os.path.basename(piece_dir.rstrip("/"))
    engraving_id = _pick_engraving(piece_dir)
    score_dir    = os.path.join(piece_dir, "scores", engraving_id)
    mung_dir     = os.path.join(score_dir, "mung")

    # Render strip once — shared across all performances
    strip, mapping = build_strip(score_dir, max_pad_px=max_pad_px, target_raw_h=target_raw_h)

    if all_performances:
        perfs = list_performances(mung_dir)
        if not perfs:
            raise ValueError(f"no performances for {piece_id}")
        # Store the canonical strip in a shared subdir at dataset root
        shared_dir = os.path.join(os.path.dirname(out_dir), "_strips")
        os.makedirs(shared_dir, exist_ok=True)
        strip_path = os.path.join(shared_dir, f"{piece_id}.png")
        if not os.path.exists(strip_path):
            strip.save(strip_path)
        results = []
        for perf_id in perfs:
            # Use short dir name: <piece_id>__<perf_suffix> where perf_suffix
            # strips the redundant piece_id prefix from the performance name.
            perf_suffix = perf_id[len(piece_id) + 1:] if perf_id.startswith(piece_id + "_") else perf_id
            perf_out = os.path.join(os.path.dirname(out_dir), f"{piece_id}__{perf_suffix}")
            try:
                r = _build_one_performance(
                    piece_dir, perf_out, piece_id, engraving_id,
                    score_dir, mung_dir, strip, mapping, strip_path, perf_id,
                )
                results.append(r)
            except Exception as e:
                results.append({"piece_id": piece_id, "performance_id": perf_id,
                                 "error": repr(e)})
        return results
    else:
        os.makedirs(out_dir, exist_ok=True)
        strip_path = os.path.join(out_dir, "strip.png")
        if not os.path.exists(strip_path):
            strip.save(strip_path)
        performance_id = _pick_performance(mung_dir, prefer=list(performance_prefer))
        r = _build_one_performance(
            piece_dir, out_dir, piece_id, engraving_id,
            score_dir, mung_dir, strip, mapping, strip_path, performance_id,
        )
        return [r]
