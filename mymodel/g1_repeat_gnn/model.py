"""G1 -- minimal GCN (Kipf & Welling style) for note-repeat embedding.

Plain PyTorch, no torch_geometric dependency (not installed on this cluster,
and piece graphs are small -- a few hundred to ~2000 notes -- so dense
adjacency matmul is cheap; no need for sparse-graph machinery).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_DIM = 7  # pitch_norm(1) + interval_ctx(4) + rel_onset(1) + rel_measure(1), see graph_data.py


class RepeatGCN(nn.Module):
    def __init__(self, in_dim: int = IN_DIM, hidden: int = 32, out_dim: int = 16, n_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = adj_norm @ x
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return F.normalize(x, dim=-1)
