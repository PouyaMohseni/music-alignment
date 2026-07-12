"""G1 -- inference-time repeat-candidate lookup from the trained GNN.

Produces a {column: [alternate columns]} dict in the SAME format/space as
build_repeat_alt_cols (STRIP-SCALED px), so it plugs directly into F6's
already-tested _decode_repeat_graph_snap without any changes to the decode
logic itself -- only the SOURCE of repeat-candidates changes (learned
embedding k-NN instead of exact n-gram matching).
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np
import torch

from mymodel.g1_repeat_gnn.graph_data import build_note_graph
from mymodel.g1_repeat_gnn.model import RepeatGCN

_MODEL_CACHE = {}


def load_gnn(checkpoint_path: str, device: str = 'cpu') -> RepeatGCN:
    if checkpoint_path not in _MODEL_CACHE:
        model = RepeatGCN()
        sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(sd['state_dict'])
        model.to(device).eval()
        _MODEL_CACHE[checkpoint_path] = model
    return _MODEL_CACHE[checkpoint_path]


@torch.no_grad()
def build_gnn_alt_cols(onset_sec: np.ndarray, midi_pitch: np.ndarray, strip_x: np.ndarray,
                       measure_idx: np.ndarray | None, w_scale: int, model: RepeatGCN,
                       sim_threshold: float = 0.85, max_candidates: int = 4,
                       min_note_distance: int = 10, device: str = 'cpu') -> dict:
    """Returns {column: [alt_cols]} in W_sc space, same convention as
    build_repeat_alt_cols. min_note_distance excludes trivially-similar
    sequential neighbors (message passing smooths adjacent-note embeddings
    together; we only want DISTANT repeat-instance candidates)."""
    N = len(midi_pitch)
    if N < 3:
        return {}
    feats, adj, _ = build_note_graph(onset_sec, midi_pitch, measure_idx)
    embed = model(feats.to(device), adj.to(device)).cpu().numpy()  # (N, D), L2-normalized

    sim = embed @ embed.T  # cosine similarity, (N, N)
    idx_grid = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    sim = np.where(idx_grid >= min_note_distance, sim, -1.0)

    cols = np.round(strip_x / w_scale).astype(np.int64)
    col_alternates = defaultdict(set)
    for i in range(N):
        cand_idx = np.argsort(-sim[i])[:max_candidates]
        for j in cand_idx:
            if sim[i, j] >= sim_threshold:
                col_alternates[int(cols[i])].add(int(cols[j]))
    return {c: sorted(v) for c, v in col_alternates.items()}
