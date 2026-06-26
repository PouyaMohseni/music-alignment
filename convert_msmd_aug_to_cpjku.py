#!/usr/bin/env python3
"""
Convert msmd_aug_v1-1_no-audio → CPJKU flat format for training.

Input:  data/MSMD/msmd_aug_v1-1_no-audio/<piece>/
Output: data/MSMD/msmd_aug_cpjku/
          score/<piece>_page_<N>.npz       (sheet, coords, coord2onset)
          performance/<piece>_page_<N>_tempo_<T>.mid
          split_all.yaml

Usage:
  python convert_msmd_aug_to_cpjku.py [--workers 8] [--out_dir PATH]

The converter reads MuNG XML annotations (which record exact notehead pixel
positions and note_event_idx per performance) to build the CPJKU-style NPZ
files, then writes per-page MIDIs derived from the pre-computed _notes.npy
feature files (which are the MSMD's authoritative note list).

Key design: we use _notes.npy as the canonical note source so that
note_event_idx values from the MuNG XML are guaranteed to be valid indices,
avoiding the parser-count-mismatch that caused IndexError in merge_onsets.
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import mido
import numpy as np
import yaml
from PIL import Image

# ── soundfont priority (determines which _notes.npy is used per tempo) ────────
SF_PRIORITY = [
    'grand-piano-YDP-20160804',
    'acoustic_piano_imis_1',
    'YamahaGrandPiano',
    'ElectricPiano',
]


def parse_mung_page(xml_path: str, piece_name: str):
    """Parse a MuNG XML page; return list of notehead dicts sorted by note_event_idx."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    noteheads = []
    for obj in root.iter('CropObject'):
        cls = obj.findtext('ClassName', '')
        if 'notehead' not in cls.lower():
            continue
        top  = float(obj.findtext('Top', 0))
        left = float(obj.findtext('Left', 0))
        w    = float(obj.findtext('Width', 1))
        h    = float(obj.findtext('Height', 1))

        idx = None
        for item in obj.iter('DataItem'):
            key = item.get('key', '')
            if key.endswith('_note_event_idx'):
                try:
                    idx = int(item.text)
                    break
                except (TypeError, ValueError):
                    continue
        if idx is None:
            continue

        noteheads.append({'top': top, 'left': left, 'w': w, 'h': h, 'note_idx': idx})

    noteheads.sort(key=lambda x: x['note_idx'])
    return noteheads


def load_page_image_gray(png_path: str) -> np.ndarray:
    """Load RGBA score page image → grayscale uint8."""
    img = Image.open(png_path).convert('RGBA')
    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return np.array(bg.convert('L'), dtype=np.uint8)


def find_best_notes_npy(perf_dir: str, piece_name: str, tempo: str):
    """Return path to the .flac_notes.npy for this tempo, using SF priority."""
    for sf in SF_PRIORITY:
        perf_id = f'{piece_name}_tempo-{tempo}_{sf}'
        feat_dir = os.path.join(perf_dir, perf_id, 'features')
        notes_path = os.path.join(feat_dir, f'{perf_id}.flac_notes.npy')
        if os.path.exists(notes_path):
            return notes_path
    return None


def write_page_midi_from_notes(notes_rows, out_path: str, ticks_per_beat: int = 480):
    """Write a page MIDI from rows of _notes.npy: [onset_sec, pitch, dur_sec, vel, ch].

    Uses a fixed 120 BPM template so onset_sec maps unambiguously to ticks.
    FluidSynth will synthesize audio with exactly these onset times in seconds.
    """
    TEMPO_US = 500000  # 120 BPM = 500 000 μs/beat

    def sec_to_ticks(s):
        return max(0, int(round(s * ticks_per_beat * 1e6 / TEMPO_US)))

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=TEMPO_US, time=0))

    events = []
    for row in notes_rows:
        onset_sec, pitch, dur_sec, vel = float(row[0]), int(row[1]), float(row[2]), int(row[3])
        onset_t  = sec_to_ticks(onset_sec)
        offset_t = sec_to_ticks(onset_sec + max(dur_sec, 0.05))  # min 50 ms
        vel = max(1, min(127, vel))
        events.append((onset_t,  'note_on',  pitch, vel))
        events.append((offset_t, 'note_off', pitch, 0))

    # note_off before note_on at the same tick (standard MIDI convention):
    # ensures adjacent same-pitch notes don't get zero duration and dropped by madmom
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'note_off' else 1))

    prev = 0
    for abs_t, mtype, pitch, vel in events:
        delta = abs_t - prev
        track.append(mido.Message(mtype, note=pitch, velocity=vel, time=max(0, delta)))
        prev = abs_t

    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(out_path)


def convert_piece(args):
    """Worker: convert one piece. Returns list of output page-names created."""
    piece_dir, out_dir = args
    piece_name = os.path.basename(piece_dir)
    score_dir = None
    for d in os.listdir(os.path.join(piece_dir, 'scores')):
        score_dir = os.path.join(piece_dir, 'scores', d)
        break
    if score_dir is None:
        return []

    img_dir   = os.path.join(score_dir, 'img')
    mung_dir  = os.path.join(score_dir, 'mung')
    perf_dir  = os.path.join(piece_dir, 'performances')
    out_score = os.path.join(out_dir, 'score')
    out_perf  = os.path.join(out_dir, 'performance')

    page_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if not page_files:
        return []

    # Discover available tempos
    tempo_set = set()
    for d in os.listdir(perf_dir):
        if '_tempo-' in d:
            tempo_set.add(d.split('_tempo-')[1].split('_')[0])
    if not tempo_set:
        return []

    # Pre-load _notes.npy per tempo (authoritative MSMD note list)
    notes_cache = {}  # tempo -> np.ndarray (N, 5)
    for tempo in tempo_set:
        npy_path = find_best_notes_npy(perf_dir, piece_name, tempo)
        if npy_path:
            try:
                notes_cache[tempo] = np.load(npy_path)
            except Exception:
                pass

    if not notes_cache:
        return []

    # Use first available tempo's note count as the authoritative N
    ref_N = len(next(iter(notes_cache.values())))

    created = []

    for page_idx, png_path in enumerate(page_files):
        page_num = os.path.splitext(os.path.basename(png_path))[0]
        mung_path = os.path.join(mung_dir, f'{page_num}.xml')
        if not os.path.exists(mung_path):
            continue

        noteheads = parse_mung_page(mung_path, piece_name)
        if not noteheads:
            continue

        # Filter noteheads whose note_event_idx is out of bounds for the MSMD note list
        noteheads = [nh for nh in noteheads if nh['note_idx'] < ref_N]
        if not noteheads:
            continue

        # ── build NPZ ───────────────────────────────────────────────────────
        sheet = load_page_image_gray(png_path)

        coords = np.array(
            [[nh['top'] + nh['h'] / 2, nh['left'] + nh['w'] / 2, nh['h'] / 2]
             for nh in noteheads],
            dtype=np.float32
        )

        global_indices = [nh['note_idx'] for nh in noteheads]
        unique_global  = sorted(set(global_indices))
        g2l = {g: l for l, g in enumerate(unique_global)}

        c2o = {i: g2l[nh['note_idx']] for i, nh in enumerate(noteheads)}
        coord2onset = np.empty(1, dtype=object)
        coord2onset[0] = c2o

        out_npz = os.path.join(out_score, f'{piece_name}_page_{page_idx}.npz')
        np.savez(out_npz, sheet=sheet, coords=coords, coord2onset=coord2onset)

        # ── write per-tempo MIDIs from _notes.npy rows ──────────────────────
        page_global_set = set(unique_global)

        for tempo, notes_npy in notes_cache.items():
            # Extract rows for this page in page-local order
            page_rows = [notes_npy[g] for g in unique_global if g < len(notes_npy)]
            if not page_rows:
                continue
            out_mid = os.path.join(out_perf, f'{piece_name}_page_{page_idx}_tempo_{tempo}.mid')
            try:
                write_page_midi_from_notes(page_rows, out_mid)
            except Exception:
                pass

        created.append(f'{piece_name}_page_{page_idx}')

    return created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='data/MSMD/msmd_aug_v1-1_no-audio')
    parser.add_argument('--out_dir', default='data/MSMD/msmd_aug_cpjku')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    (out / 'score').mkdir(parents=True, exist_ok=True)
    (out / 'performance').mkdir(parents=True, exist_ok=True)

    pieces = sorted([
        str(p) for p in src.iterdir()
        if p.is_dir() and (p / 'scores').exists() and (p / 'performances').exists()
    ])
    print(f'Found {len(pieces)} pieces to convert')

    work = [(p, str(out)) for p in pieces]
    all_names = []
    with Pool(args.workers) as pool:
        for i, names in enumerate(pool.imap_unordered(convert_piece, work), 1):
            all_names.extend(names)
            if i % 50 == 0:
                print(f'  {i}/{len(pieces)} pieces done, {len(all_names)} pages so far')

    all_names.sort()
    with open(out / 'split_all.yaml', 'w') as f:
        yaml.dump({'files': all_names}, f)

    print(f'\nDone. {len(all_names)} score pages written to {out}/')
    print(f'Split file: {out}/split_all.yaml')


if __name__ == '__main__':
    main()
