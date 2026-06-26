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
files, then extracts per-page MIDIs from the full-piece performance MIDIs.
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

# ── soundfont priority (determines which MIDI is used per tempo) ───────────────
# Grand-piano covers 500,750,1000,1250,1500,1750,2000; others cover 900,950,1050,1100
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

        # Extract note_event_idx from any available performance key
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
            continue  # notehead not linked to any MIDI note

        noteheads.append({'top': top, 'left': left, 'w': w, 'h': h, 'note_idx': idx})

    # Sort by note index (temporal/reading order)
    noteheads.sort(key=lambda x: x['note_idx'])
    return noteheads


def load_page_image_gray(png_path: str) -> np.ndarray:
    """Load RGBA score page image → grayscale uint8 (same dtype as Zenodo NPZ)."""
    img = Image.open(png_path).convert('RGBA')
    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return np.array(bg.convert('L'), dtype=np.uint8)


def midi_to_note_list(midi_path: str):
    """Return (notes, ticks_per_beat, tempo_msgs).
    notes: list of (onset_ticks, offset_ticks, pitch, velocity) sorted by onset.
    tempo_msgs: list of (abs_tick, tempo_us) for preserving original tempo.
    """
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat

    all_msgs = []
    tempo_msgs = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_msgs.append((abs_tick, msg.tempo))
            elif msg.type in ('note_on', 'note_off'):
                all_msgs.append((abs_tick, msg.type, msg.note, msg.velocity))

    all_msgs.sort(key=lambda x: x[0])

    open_notes = {}
    notes = []
    for abs_tick, msg_type, pitch, velocity in all_msgs:
        if msg_type == 'note_on' and velocity > 0:
            open_notes[pitch] = (abs_tick, velocity)
        else:  # note_off or note_on vel=0
            if pitch in open_notes:
                onset, vel = open_notes.pop(pitch)
                notes.append((onset, abs_tick, pitch, vel))

    notes.sort(key=lambda x: x[0])
    return notes, tpb, tempo_msgs


def write_page_midi(notes_subset, start_tick, ticks_per_beat, tempo_msgs, out_path: str):
    """Write a MIDI file containing only notes_subset, time-shifted to start at 0."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Preserve original tempo events (shifted)
    for abs_tick, tempo in tempo_msgs:
        shifted = max(0, abs_tick - start_tick)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=int(shifted)))

    # Build sorted event list
    events = []
    for onset, offset, pitch, velocity in notes_subset:
        events.append((onset - start_tick, 'note_on',  pitch, velocity))
        events.append((offset - start_tick, 'note_off', pitch, 0))
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'note_off' else 1))

    prev = 0
    for abs_t, mtype, pitch, vel in events:
        delta = int(abs_t) - prev
        track.append(mido.Message(mtype, note=pitch, velocity=vel, time=max(0, delta)))
        prev = int(abs_t)

    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(out_path)


def find_best_midi(perf_dir: str, piece_name: str, tempo: str) -> str | None:
    """Return path to MIDI for given tempo using soundfont priority, or None."""
    for sf in SF_PRIORITY:
        perf_id = f'{piece_name}_tempo-{tempo}_{sf}'
        midi_path = os.path.join(perf_dir, perf_id, f'{perf_id}.midi')
        if os.path.exists(midi_path):
            return midi_path
    return None


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

    # Discover available tempos across all performance dirs
    tempo_set = set()
    for d in os.listdir(perf_dir):
        if '_tempo-' in d:
            tempo = d.split('_tempo-')[1].split('_')[0]
            tempo_set.add(tempo)
    if not tempo_set:
        return []

    # Pre-load all piece MIDIs (one per tempo) to avoid re-loading per page
    midi_cache = {}   # tempo -> (notes, tpb, tempo_msgs)
    for tempo in tempo_set:
        midi_path = find_best_midi(perf_dir, piece_name, tempo)
        if midi_path:
            try:
                midi_cache[tempo] = midi_to_note_list(midi_path)
            except Exception:
                pass

    if not midi_cache:
        return []

    created = []

    for page_idx, png_path in enumerate(page_files):
        page_num = os.path.splitext(os.path.basename(png_path))[0]  # "01", "02", ...
        mung_path = os.path.join(mung_dir, f'{page_num}.xml')
        if not os.path.exists(mung_path):
            continue

        noteheads = parse_mung_page(mung_path, piece_name)
        if not noteheads:
            continue

        # Get note count from first MIDI to filter out-of-range note_event_idx values.
        # MuNG annotations sometimes reference notes beyond the MIDI's actual length.
        ref_notes, _, _ = next(iter(midi_cache.values()))
        n_midi_notes = len(ref_notes)
        noteheads = [nh for nh in noteheads if nh['note_idx'] < n_midi_notes]
        if not noteheads:
            continue

        # ── build NPZ ───────────────────────────────────────────────────────
        sheet = load_page_image_gray(png_path)

        # coords: (N, 3) = [y_center, x_center, h/2]  (float32, like Zenodo)
        coords = np.array(
            [[nh['top'] + nh['h'] / 2, nh['left'] + nh['w'] / 2, nh['h'] / 2]
             for nh in noteheads],
            dtype=np.float32
        )

        # Global note indices for this page
        global_indices = [nh['note_idx'] for nh in noteheads]
        unique_global = sorted(set(global_indices))
        g2l = {g: l for l, g in enumerate(unique_global)}  # global→page-local

        # coord2onset: {coord_idx: page_local_note_idx}  (stored as 1-elem object array)
        c2o = {i: g2l[nh['note_idx']] for i, nh in enumerate(noteheads)}
        coord2onset = np.empty(1, dtype=object)
        coord2onset[0] = c2o

        out_npz = os.path.join(out_score, f'{piece_name}_page_{page_idx}.npz')
        np.savez(out_npz, sheet=sheet, coords=coords, coord2onset=coord2onset)

        # ── write per-tempo MIDIs ────────────────────────────────────────────
        page_global_set = set(global_indices)

        for tempo, (all_notes, tpb, tempo_msgs) in midi_cache.items():
            page_notes = [n for i, n in enumerate(all_notes) if i in page_global_set]
            if not page_notes:
                continue
            start_tick = page_notes[0][0]
            out_mid = os.path.join(out_perf, f'{piece_name}_page_{page_idx}_tempo_{tempo}.mid')
            try:
                write_page_midi(page_notes, start_tick, tpb, tempo_msgs, out_mid)
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
