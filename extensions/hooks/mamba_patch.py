"""Swap CYOLO's recurrent core for CODA's Mamba, and change nothing else.

WHY THIS IS THE ABLATION CODA DID NOT RUN
-----------------------------------------
CODA's Table 3 ablates the cascade, cross-attention, beam search, the temporal
priors and scheduled sampling. It never ablates the audio encoder, so the paper
does not show that its Mamba tower contributes anything at all. The two models
make that testable, because their conditioning paths have the same shape:

    CYOLO-SB   78x40 log-mel CNN -> 32 -> LSTM(32 -> 64), 1 layer -> z (128)
    CODA       audio             ->       Mamba(64), 2 layers     -> z (128)

Same width, same depth of projection, same 128-dimensional conditioning vector.
So `seq_model` is the whole difference and it is a drop-in swap: the CNN front
end, the kw=40 window, the concat with the last step, the FPN, the anchors and
the multi-class head are all untouched, and a change in the result is
attributable to the recurrence and to nothing else.

This is the opposite half of the architecture from H1. H1 replaced the CNN with
MERT and cost 40 points on room; this leaves the CNN alone and replaces what
reads its output over time.

WHY IT IS INSTALLED AFTER __init__
----------------------------------
`Model.__init__` finishes with `self.apply(initialize_weights)`, which
orthogonalises every Linear and zeroes every bias. Mamba's behaviour depends on
its own initialisation of A_log, D and the dt projection bias, and orthogonal
weights with zero bias would destroy the timescale prior that makes the block
train at all. So the swap happens after the model is built and initialised, and
the new block keeps the initialisation its authors chose.

THE COMPARISON HAS TO BE BUDGET MATCHED
---------------------------------------
Our detector trainings reach about ten epochs in a 24 hour A100 job, nowhere
near Henkel's released checkpoint. A Mamba run measured against that release
would confound the recurrence with three orders of magnitude of compute, so the
LSTM control is retrained here under the identical budget, seed and data, and
only the two of them are compared with each other.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class MambaSeq(nn.Module):
    """A stand-in for nn.LSTM in ContextConditioning.encode_sequence.

    The caller does `_, hidden = self.seq_model(packed, hidden)` and then reads
    `hidden[0][-1]` as the sequence summary, so the return shape is what has to
    match, not the type. There is no streaming state to carry: cyolo rebuilds a
    kw=40 window every step and calls encode_sequence with hidden=None, which is
    why a non-recurrent-in-time wrapper is faithful here.
    """

    def __init__(self, in_dim=32, hidden_size=64, n_layers=2, d_state=16,
                 d_conv=4, expand=2):
        super().__init__()
        from mamba_ssm import Mamba
        self.hidden_size = hidden_size
        self.inp = nn.Linear(in_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [Mamba(d_model=hidden_size, d_state=d_state, d_conv=d_conv,
                   expand=expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_layers)])

    def forward(self, packed, hidden=None):
        x, lengths = pad_packed_sequence(packed, batch_first=True)
        h = self.inp(x)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))
        idx = (lengths - 1).clamp(min=0).to(h.device)
        last = h[torch.arange(h.shape[0], device=h.device), idx]
        # (n_layers, B, hidden) so that hidden[0][-1] is the summary, exactly
        # what the LSTM path hands to z_enc
        s = last.unsqueeze(0)
        return None, (s, s)


def patch_mamba(n_layers=2, d_state=16, d_conv=4, expand=2):
    from cyolo_score_following.models.yolo import Model

    if getattr(Model, '_mamba_patched', False):
        return
    prev_init = Model.__init__

    def __init__(self, *a, **kw):
        prev_init(self, *a, **kw)
        cn = self.conditioning_network
        old = cn.seq_model
        in_dim = old.input_size
        hidden = old.hidden_size
        cn.seq_model = MambaSeq(in_dim=in_dim, hidden_size=hidden,
                                n_layers=n_layers, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        n_old = sum(p.numel() for p in old.parameters())
        n_new = sum(p.numel() for p in cn.seq_model.parameters())
        print(f'[MAMBA] seq_model LSTM({in_dim}->{hidden}, {n_old} params) -> '
              f'Mamba x{n_layers} (d_state={d_state}, d_conv={d_conv}, '
              f'expand={expand}, {n_new} params)', flush=True)

    Model.__init__ = __init__
    Model._mamba_patched = True


def maybe_patch_mamba() -> bool:
    if os.environ.get('MAMBA_ENC', '0') == '0':
        return False
    patch_mamba(n_layers=int(os.environ.get('MAMBA_LAYERS', '2')),
                d_state=int(os.environ.get('MAMBA_DSTATE', '16')),
                d_conv=int(os.environ.get('MAMBA_DCONV', '4')),
                expand=int(os.environ.get('MAMBA_EXPAND', '2')))
    return True
