#!/usr/bin/env python3
"""Fill missing per-tempo MIDIs in msmd_aug_cpjku/performance/ so all 11 tempos
exist for every page in split_all.yaml.

Strategy: existing page MIDIs use a fixed 120 BPM (TEMPO_US=500000) base with
note onsets encoded as tick positions. Changing only the MIDI tempo event scales
all onset times proportionally — no tick modification needed.

For a page missing tempo-900:
  new_tempo_us = 500000 * (900 / 1000)   # relative to reference tempo 1000
This makes the audio play at 0.9x speed = 900 ms/beat. Same notes, correct timing.

Usage:
    python fill_missing_tempo_midis.py [--data PATH] [--ref_tempo 1000] [--dry_run]
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import mido
import yaml

TEMPOS = [500, 750, 900, 950, 1000, 1050, 1100, 1250, 1500, 1750, 2000]
REF_TEMPO_US = 500000  # the base used by write_page_midi_from_notes (120 BPM)


def _scale_midi_tempo(src_path: str, target_ms: int, out_path: str):
    """Copy src MIDI, replacing the tempo event so onset times scale to target_ms/beat."""
    mid = mido.MidiFile(src_path)
    new_us = int(REF_TEMPO_US * target_ms / 1000)
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)
        replaced = False
        for msg in track:
            if msg.type == 'set_tempo':
                if not replaced:
                    new_track.append(
                        mido.MetaMessage('set_tempo', tempo=new_us, time=msg.time))
                    replaced = True
                # drop extra tempo events
            else:
                new_track.append(msg.copy())
        if not replaced and new_mid.tracks[0] is new_track:
            new_track.insert(0, mido.MetaMessage('set_tempo', tempo=new_us, time=0))
    new_mid.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/scratch/pmohseni/music-alignment/msmd_aug_cpjku')
    ap.add_argument('--ref_tempo', type=int, default=1000,
                    help='Tempo to use as reference for scaling (must be complete)')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    data = Path(args.data)
    split_file = data / 'split_all.yaml'

    with open(split_file) as f:
        orig = yaml.safe_load(f)

    pages = orig['files']
    needed = filled = skipped = failed = 0

    for page in pages:
        ref_mid = data / 'performance' / f'{page}_tempo_{args.ref_tempo}.mid'
        if not ref_mid.exists():
            print(f"WARN: reference tempo_{args.ref_tempo} missing for {page}, skipping")
            continue

        for t in TEMPOS:
            out_path = data / 'performance' / f'{page}_tempo_{t}.mid'
            if out_path.exists():
                skipped += 1
                continue
            needed += 1
            if args.dry_run:
                print(f"  [DRY] would create {out_path.name}")
                continue
            try:
                _scale_midi_tempo(str(ref_mid), t, str(out_path))
                filled += 1
            except Exception as e:
                print(f"  FAIL {page} tempo_{t}: {e}")
                failed += 1

    print(f"\nDone. needed={needed}  filled={filled}  skipped={skipped}  failed={failed}")
    if not args.dry_run and filled > 0:
        # Verify all pages now have all 11 tempos
        still_missing = sum(
            1 for page in pages
            for t in TEMPOS
            if not (data / 'performance' / f'{page}_tempo_{t}.mid').exists()
        )
        print(f"Pages still missing after fill: {still_missing}")


if __name__ == '__main__':
    main()
