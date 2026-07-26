"""M1 model (Phase 1). Deliberately D1 + a single delta: instead of a dense
frame x SPATIAL-column similarity matrix trained with per-frame CE, produce a
frame x ONSET-column alignment matrix trained with the forward-sum monotonic
objective (extensions/alignment/forward_sum.py). Reuses D1's proven,
strip-format-consistent MERT audio tower and CNN+transformer score tower
verbatim (mymodel/d1_align_matrix/model.py) -- the ONLY change is which
columns the audio attends to and how the matrix is supervised.

Why onset columns (not D1's uniform spatial columns): the forward-sum path
model is stay-or-advance-by-one and surjective (every column visited once, in
order), which is exactly the structure of the alignment when columns are the
piece's ordered onset anchors -- each onset advances the path by one, frames
between onsets stay. Uniform spatial columns would need a different,
looser DP (advance-by-any), weakening the monotonic inductive bias. Onset
columns also match how eval measures error (at onset times).

The similarity S = A @ B_onset^T / temperature is single-head cross-attention
scoring (audio-frame query . score-onset-column key); the QK^T scores ARE the
alignment matrix the forward-sum consumes -- no softmax/value step is needed to
produce the matrix (that lives inside the loss). Multi-head / learned-projection
cross-attention is a Phase-2 ablation, not needed to prove the objective trains.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from mymodel.d1_align_matrix.model import AudioTower, ScoreTower


class M1Model(nn.Module):
    def __init__(self, d_mert: int = 768, d_model: int = 128, w_downsample: int = 4,
                 n_ctx_layers: int = 2, n_heads: int = 4, temperature: float = 0.07,
                 score_tower: str = 'cnn'):
        super().__init__()
        self.audio_tower = AudioTower(d_mert, d_model)
        self.score_tower_kind = score_tower
        if score_tower == 'musvit':
            from mymodel.m1_monotonic.musvit_tower import MuSViTScoreTower
            self.score_tower = MuSViTScoreTower(d_model, n_ctx_layers, n_heads)
        elif score_tower == 'dinov2':
            from mymodel.m1_monotonic.dinov2_tower import Dinov2ScoreTower
            self.score_tower = Dinov2ScoreTower(d_model, n_ctx_layers, n_heads)
        else:
            self.score_tower = ScoreTower(d_model, w_downsample, n_ctx_layers, n_heads)
        self.temperature = temperature
        self.w_downsample = w_downsample

    def forward(self, mert: torch.Tensor, strip: torch.Tensor,
                onset_spatial_cols: torch.Tensor) -> torch.Tensor:
        """mert: (T, d_mert). strip: (1, 1, H, W). onset_spatial_cols: (N,) long,
        the spatial-column index of each onset anchor (into the score tower's
        W_col output). Returns S: (T, N) alignment scores over onset columns."""
        A = self.audio_tower(mert)              # (T, d), L2-normalized per frame
        B = self.score_tower(strip)             # (W_col, d), L2-normalized per column
        W_col = B.shape[0]
        cols = onset_spatial_cols.clamp(min=0, max=W_col - 1)
        B_onset = B.index_select(0, cols)       # (N, d)
        return (A @ B_onset.t()) / self.temperature   # (T, N)
