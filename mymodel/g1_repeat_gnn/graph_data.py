"""G1 -- score-only note graph construction for the repeat-embedding GNN.

Builds, per piece, a small graph over notes (nodes) with sequential-adjacency
edges (melodic context for message passing) and separately extracts
heuristic repeat-equivalent NOTE-INDEX groups (weak/noisy contrastive
training targets, reusing D2's transposition-invariant pitch-interval n-gram
key unchanged). Repeat groups are NOT fed as message-passing edges -- only
as the training signal -- so the GNN has to learn to recognize repeat
structure from local melodic context alone (a genuine generalization task),
rather than trivially satisfying the loss by aggregating over the exact
edges it's asked to predict.

Score-only: no audio, no ground-truth alignment position used anywhere here
-- same legitimacy bar as build_repeat_alt_cols (this is information a
performer reading the printed score already has in advance).
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np
import torch

from mymodel.d2_midi_privileged.repeat_labels import _interval_ngram_key


def find_repeat_note_groups(pitches: np.ndarray, k: int = 5) -> list[list[int]]:
    """pitches: (N,) MIDI pitch per note, in onset order. Returns a list of
    NOTE-INDEX groups (each >=2 members) that share an identical/transposed
    local pitch-interval context ending at that note -- same construction as
    find_repeat_groups, but returning note indices directly instead of
    column positions (graph nodes ARE notes, not columns)."""
    N = len(pitches)
    groups = defaultdict(list)
    for i in range(k - 1, N):
        window = pitches[i - k + 1:i + 1].tolist()
        key = _interval_ngram_key(window)
        groups[key].append(i)
    return [idxs for idxs in groups.values() if len(idxs) >= 2]


def build_note_graph(onset_sec: np.ndarray, midi_pitch: np.ndarray,
                     measure_idx: np.ndarray | None = None, k: int = 5,
                     context_window: int = 4):
    """Returns (features: (N, D) float32, adj_norm: (N, N) float32 dense
    symmetric-normalized adjacency with self-loops, repeat_groups: list of
    note-index lists). Notes are assumed pre-sorted by onset (as noteheads.npz
    guarantees by construction)."""
    N = len(midi_pitch)
    pitch = midi_pitch.astype(np.float32)
    pitch_norm = (pitch - 60.0) / 24.0  # centered on middle C, ~2 octave scale

    # Local transposition-invariant interval context (same window D2's
    # heuristic uses) as an explicit feature -- gives the GNN a computed cue
    # to refine rather than forcing it to rediscover interval arithmetic
    # purely from raw pitch via message passing.
    interval_ctx = np.zeros((N, context_window), dtype=np.float32)
    for i in range(N):
        lo = max(0, i - context_window)
        deltas = np.diff(pitch[lo:i + 1])
        interval_ctx[i, -len(deltas):] = deltas if len(deltas) else 0.0
    interval_ctx = np.clip(interval_ctx / 12.0, -2.0, 2.0)  # scale to ~[-2,2]

    dur = max(float(onset_sec[-1] - onset_sec[0]), 1e-6)
    rel_onset = ((onset_sec - onset_sec[0]) / dur).astype(np.float32)

    if measure_idx is not None:
        m = measure_idx.astype(np.float32)
        rel_measure = (m - m.min()) / max(float(m.max() - m.min()), 1.0)
    else:
        rel_measure = rel_onset.copy()

    features = np.concatenate([
        pitch_norm[:, None], interval_ctx, rel_onset[:, None], rel_measure[:, None],
    ], axis=1).astype(np.float32)

    # Sequential adjacency only (bidirectional) + self-loops, symmetric-normalized
    # (Kipf & Welling GCN convention: D^-1/2 (A+I) D^-1/2).
    A = np.eye(N, dtype=np.float32)
    idx = np.arange(N - 1)
    A[idx, idx + 1] = 1.0
    A[idx + 1, idx] = 1.0
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-6))
    adj_norm = (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]

    repeat_groups = find_repeat_note_groups(midi_pitch, k=k)

    return (torch.from_numpy(features), torch.from_numpy(adj_norm.astype(np.float32)),
            repeat_groups)
