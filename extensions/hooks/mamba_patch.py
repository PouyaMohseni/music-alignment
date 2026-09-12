"""CODA's Mamba audio encoder in CYOLO, in the slot MERT occupied.

WHAT CODA ACTUALLY DOES (Section 3.2)
-------------------------------------
    "The audio stream is converted into a 78-dimensional log-filterbank at 20
    frames per second. A two-layer causal Mamba encoder processes each frame.
    Its recurrent state compactly accumulates the entire audio history up to
    frame t, yielding a conditioning vector z_t."

There is no CNN in their audio path. Mamba is the whole encoder, taking the
filterbank frame by frame. Their implementation details: two layers, hidden 64,
state 16, convolution width 4, expansion 2, into a 128-dimensional z.

CYOLO-SB's audio path, for the same 78-bin input and the same 128-d output:

    78 x 40 window -> 2D CNN -> 32 per block -> LSTM(32 -> 64), 1 layer
                   -> concat with the encoded last window -> z (128)

So the swap is the entire ContextConditioning audio tower, which is the same
slot H1 put MERT into. Two encoders, opposite ends of the design space -- a
768-d pretrained transformer and a 64-wide recurrence over raw filterbank --
against the same detector, the same head and the same 128-d conditioning
interface. If neither moves real-audio accuracy while reranking the very same
candidates moves it by twenty points, perception is not the constraint.

WHAT IS DELIBERATELY NOT PORTED
-------------------------------
CODA also keeps a sliding window H_t of recent Mamba outputs as keys and values
for cross-attention against candidate regions. That belongs to their cascade,
not to their encoder, and we have no cross-attention to feed. Porting it would
make this a partial reimplementation of CODA rather than an encoder swap.

COST, WHICH HAS TO BE MEASURED NOT ASSUMED
------------------------------------------
CYOLO's CNN chops the history into 40-frame blocks, so its recurrence takes T/40
steps. A per-frame Mamba takes T, forty times as many, and cyolo's data pipeline
re-encodes the whole history at every frame rather than streaming it. The
smoke test times both arms so the budget-matched comparison can be set up on
measured throughput instead of a guess.

WHY IT IS INSTALLED AFTER __init__
----------------------------------
`Model.__init__` ends with `self.apply(initialize_weights)`, which
orthogonalises every Linear and zeroes every bias. Mamba's behaviour depends on
its own initialisation of A_log, D and the dt projection bias, and that would
destroy the timescale prior the block needs to train.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_mamba():
    """Import the Mamba block past its own package init.

    mamba_ssm 2.2.4's `__init__` imports MambaLMHeadModel, whose generation
    utils do `from transformers.generation import GreedySearchDecoderOnlyOutput,
    SampleDecoderOnlyOutput`. transformers removed both in v5 (5.3.0 here), so
    the import fails. Reaching for the submodule directly does not help:
    importing any `mamba_ssm.x` runs the package `__init__` first.

    Downgrading transformers would fix it and would also silently change the
    library the MERT banks were built against, so instead the two names are
    stubbed. Nothing in the selective-scan block touches generation; the
    symbols only have to exist for the import statement to bind them. This is
    the same tactic as madmom_compat.patch() elsewhere in this repo.
    """
    import transformers.generation as tg
    for name in ('GreedySearchDecoderOnlyOutput', 'SampleDecoderOnlyOutput',
                 'GreedySearchEncoderDecoderOutput', 'SampleEncoderDecoderOutput'):
        if not hasattr(tg, name):
            setattr(tg, name, type(name, (), {}))
    from mamba_ssm.modules.mamba_simple import Mamba
    return Mamba


class MambaAudioEncoder(nn.Module):
    """Drop-in for ContextConditioning: 78-bin frames in, a 128-d z out.

    Only `encode_sequence` is required. cyolo calls it from Model.forward with
    hidden=None and reads element 0; `get_conditioning` is used by test.py
    alone, which nothing in our pipeline runs.
    """

    def __init__(self, n_mels=78, hidden_size=64, zdim=128, n_layers=2,
                 d_state=16, d_conv=4, expand=2, max_hist=0):
        super().__init__()
        Mamba = _import_mamba()
        self.hidden_size, self.zdim, self.max_hist = hidden_size, zdim, max_hist
        self.inp = nn.Linear(n_mels, hidden_size)
        # named seq_model because iterate_dataset clips gradients on
        # `conditioning_network.seq_model.parameters()`, and the recurrence is
        # what that clip is for in the LSTM path too
        self.seq_model = nn.ModuleList(
            [Mamba(d_model=hidden_size, d_state=d_state, d_conv=d_conv,
                   expand=expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_layers)])
        self.z_enc = nn.Sequential(nn.Linear(hidden_size, zdim),
                                   nn.LayerNorm(zdim), nn.ELU())
        # cyolo's ContextConditioning carries these and the dataset reads them
        self.kw, self.kh = 40, n_mels
        self.dw, self.dh = 1, 1

    def encode_sequence(self, x, hidden=None):
        """x: list of (T_i, n_mels) spectrograms, one per batch item."""
        if self.max_hist:
            x = [s[-self.max_hist:] for s in x]
        lengths = torch.tensor([max(1, s.shape[0]) for s in x])
        T = int(lengths.max())
        dev = x[0].device
        padded = torch.stack([F.pad(s, (0, 0, 0, T - s.shape[0])) if s.shape[0] < T
                              else s[:T] for s in x])
        h = self.inp(padded)
        for blk, nrm in zip(self.seq_model, self.norms):
            h = h + blk(nrm(h))
        idx = (lengths - 1).to(dev)
        last = h[torch.arange(h.shape[0], device=dev), idx]
        z = self.z_enc(last)
        # (n_layers, B, hidden) so a caller reading hidden[0][-1] still works
        s = last.unsqueeze(0)
        return z, (s, s)

    def encode_samples(self, x):
        raise NotImplementedError('the Mamba tower has no block structure')


class MambaSeq(nn.Module):
    """Mamba in place of the LSTM ALONE, keeping CYOLO's convolutional front end.

    This is not CODA's design. It exists to decompose the tower swap: the CODA
    arm changes the frame encoder and the recurrence at once, so on its own it
    cannot say which half matters. With this arm the pair reads

        CNN   + LSTM     the released architecture
        CNN   + Mamba    recurrence only
        Mamba            CODA's, both at once
        MERT  + LSTM     frame encoder only (H1)

    and the two middle rows separate what the outer two confound.

    The caller does `_, hidden = self.seq_model(packed, hidden)` and reads
    `hidden[0][-1]`, so the return shape is what has to match, not the type.
    """

    def __init__(self, in_dim=32, hidden_size=64, n_layers=2, d_state=16,
                 d_conv=4, expand=2):
        super().__init__()
        Mamba = _import_mamba()
        self.hidden_size = hidden_size
        self.inp = nn.Linear(in_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [Mamba(d_model=hidden_size, d_state=d_state, d_conv=d_conv,
                   expand=expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_layers)])

    def forward(self, packed, hidden=None):
        from torch.nn.utils.rnn import pad_packed_sequence
        x, lengths = pad_packed_sequence(packed, batch_first=True)
        h = self.inp(x)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))
        idx = (lengths - 1).clamp(min=0).to(h.device)
        last = h[torch.arange(h.shape[0], device=h.device), idx]
        s = last.unsqueeze(0)
        return None, (s, s)


def patch_mamba(n_layers=2, d_state=16, d_conv=4, expand=2, max_hist=0,
                mode='tower'):
    from cyolo_score_following.models.yolo import Model

    if getattr(Model, '_mamba_patched', False):
        return
    prev_init = Model.__init__

    def __init__(self, *a, **kw):
        prev_init(self, *a, **kw)
        old = self.conditioning_network
        if mode == 'seq':
            lstm = old.seq_model
            old.seq_model = MambaSeq(in_dim=lstm.input_size,
                                     hidden_size=lstm.hidden_size,
                                     n_layers=n_layers, d_state=d_state,
                                     d_conv=d_conv, expand=expand)
            n_o = sum(p.numel() for p in lstm.parameters())
            n_n = sum(p.numel() for p in old.seq_model.parameters())
            print(f'[MAMBA] recurrence only: LSTM({lstm.input_size}->'
                  f'{lstm.hidden_size}, {n_o} params) -> Mamba x{n_layers} '
                  f'({n_n} params), CNN front end kept', flush=True)
            return
        n_mels = getattr(old, 'kh', 78)
        hid = getattr(old.seq_model, 'hidden_size', 64)
        new = MambaAudioEncoder(n_mels=n_mels, hidden_size=hid,
                                zdim=self.zdim, n_layers=n_layers, d_state=d_state,
                                d_conv=d_conv, expand=expand, max_hist=max_hist)
        self.conditioning_network = new
        n_old = sum(p.numel() for p in old.parameters())
        n_new = sum(p.numel() for p in new.parameters())
        print(f'[MAMBA] audio tower CNN+LSTM ({n_old} params) -> causal Mamba '
              f'x{n_layers} on {n_mels}-bin frames (d_state={d_state}, '
              f'd_conv={d_conv}, expand={expand}, hist={max_hist or "full"}, '
              f'{n_new} params)', flush=True)

    Model.__init__ = __init__
    Model._mamba_patched = True


def maybe_patch_mamba() -> bool:
    if os.environ.get('MAMBA_ENC', '0') == '0':
        return False
    patch_mamba(n_layers=int(os.environ.get('MAMBA_LAYERS', '2')),
                d_state=int(os.environ.get('MAMBA_DSTATE', '16')),
                d_conv=int(os.environ.get('MAMBA_DCONV', '4')),
                expand=int(os.environ.get('MAMBA_EXPAND', '2')),
                max_hist=int(os.environ.get('MAMBA_MAXHIST', '0')),
                mode=os.environ.get('MAMBA_MODE', 'tower'))
    return True
