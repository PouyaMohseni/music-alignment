"""E2/E3 dataset: v13's FullStripDataset + two MIDI-privileged, train-time-
only fields per piece: `pitch_roll` (T, 88) for distillation, and
`repeat_alt_cols` ({column: [alt columns]}) for repeat-aware GT construction.
Both derived from noteheads.npz's existing onset_sec/midi_offset_sec/
midi_pitch/strip_x arrays -- no new data dependency, no MIDI file parse.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from mymodel.v13_mert_unet.data import load_piece as _v13_load_piece
from mymodel.v13_midi_privileged.repeat_gt import build_repeat_alt_cols, compute_pitch_roll_from_notes


def load_piece(piece_dir: Path, emb_root: str, h_strip: int, w_scale: int,
              fps: int = 20, repeat_k: int = 5):
    d = _v13_load_piece(piece_dir, emb_root, h_strip, w_scale, fps)
    if d is None:
        return None

    notes = np.load(piece_dir / 'noteheads.npz')
    onset_sec = notes['onset_sec'].astype(np.float64)
    offset_sec = notes['midi_offset_sec'].astype(np.float64)
    midi_pitch = notes['midi_pitch']
    strip_x = notes['strip_x'].astype(np.float32)

    d['repeat_alt_cols'] = build_repeat_alt_cols(onset_sec, midi_pitch, strip_x, w_scale, fps, k=repeat_k)
    d['pitch_roll'] = compute_pitch_roll_from_notes(onset_sec, offset_sec, midi_pitch, d['T'], fps)
    return d


class MidiPrivilegedFullStripDataset:
    def __init__(self, processed_root: str, emb_root: str, split: str,
                h_strip: int = 128, w_scale: int = 4, fps: int = 20, repeat_k: int = 5):
        self.root = Path(processed_root)
        splits = json.load(open(self.root / 'splits.json'))
        piece_ids = splits[split]

        print(f'Loading {len(piece_ids)} pieces ({split})...', flush=True)
        self.pieces = []
        n_with_repeats = 0
        for pid in piece_ids:
            d = load_piece(self.root / pid, emb_root, h_strip, w_scale, fps, repeat_k)
            if d is not None:
                self.pieces.append(d)
                if d['repeat_alt_cols']:
                    n_with_repeats += 1
            else:
                print(f'  SKIP {pid}', flush=True)
        print(f'  Loaded {len(self.pieces)}/{len(piece_ids)} pieces; '
              f'{n_with_repeats} have >=1 repeat-ambiguous column.', flush=True)

    def __len__(self): return len(self.pieces)
    def __getitem__(self, i): return self.pieces[i]
