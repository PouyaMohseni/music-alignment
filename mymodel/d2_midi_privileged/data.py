"""D2 dataset -- wraps D1's piece loader, adding two MIDI-privileged,
train-time-only fields: `pitch_roll` (T, 88) for the distillation loss, and
`repeat_alt_cols` (per-frame list of alternate columns) for the soft CE
target. Both are derived from the SAME whole-piece MIDI file already present
in cpjku_fmt/performance/<piece>.mid -- no new data dependency.

D2Piece IS-A D1Piece (same fields D1's model/eval already consume) plus the
two MIDI-only additions, so eval.py (D1's, re-exported unchanged by D2) never
needs to know about them.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from mymodel.d1_align_matrix.data import D1Piece, load_piece as d1_load_piece
from mymodel.d2_midi_privileged.midi_encoder import compute_pitch_roll
from mymodel.d2_midi_privileged.repeat_labels import build_repeat_groups_for_piece


class D2Piece(D1Piece):
    def __init__(self, d1piece: D1Piece, pitch_roll: torch.Tensor, repeat_alt_cols: list):
        self.__dict__.update(d1piece.__dict__)
        self.pitch_roll = pitch_roll                # (T, 88) float32 tensor
        self.repeat_alt_cols = repeat_alt_cols       # length-T list[list[int]]


def load_piece(piece_name, cpjku_data, mert_roots, scale_factor, w_downsample,
              repeat_k=5):
    p = d1_load_piece(piece_name, cpjku_data, mert_roots, scale_factor, w_downsample)
    if p is None:
        return None

    T = p.mert.shape[0]
    midi_path = Path(cpjku_data) / 'performance' / f'{piece_name}.mid'
    if not midi_path.exists():
        return None   # D2 requires MIDI (privileged signal); skip pieces without it

    pitch_roll = compute_pitch_roll(str(midi_path), T, fps=20)

    npz = np.load(Path(cpjku_data) / 'score' / f'{piece_name}.npz', allow_pickle=True)
    coords = npz['coords'].astype(np.float32) / scale_factor
    onset_frames = npz['onset_frames']
    col_alternates = build_repeat_groups_for_piece(
        str(midi_path), coords, onset_frames, w_downsample, k=repeat_k)

    # per-frame alt list: only onset frames carry alternates (dense_ce applies
    # every frame via interpolated GT column, but repeat ambiguity is only
    # meaningfully defined AT note onsets, where the pitch-context exists).
    repeat_alt_cols = [[] for _ in range(T)]
    onset_to_true_col = {}
    for i, f in enumerate(onset_frames):
        f = int(f)
        if 0 <= f < T:
            onset_to_true_col[f] = int(round(coords[i, 1] / w_downsample))
    for f, true_col in onset_to_true_col.items():
        if true_col in col_alternates:
            repeat_alt_cols[f] = col_alternates[true_col]

    return D2Piece(p, torch.from_numpy(pitch_roll), repeat_alt_cols)


def load_split(split, processed_root, cpjku_data, mert_roots, scale_factor,
              w_downsample, repeat_k=5, limit=None):
    import json
    splits = json.load(open(Path(processed_root) / 'splits.json'))
    names = splits.get(split, [])
    if limit is not None:
        names = names[:limit]
    pieces, skipped = [], []
    for name in names:
        p = load_piece(name, cpjku_data, mert_roots, scale_factor, w_downsample, repeat_k)
        (pieces if p is not None else skipped).append(p if p is not None else name)
    n_with_repeats = sum(1 for p in pieces if any(p.repeat_alt_cols))
    print(f'[D2Dataset] split={split}: loaded {len(pieces)}, skipped {len(skipped)} '
          f'(missing score/MERT/MIDI); {n_with_repeats} pieces have >=1 repeat-ambiguous onset',
          flush=True)
    return pieces
