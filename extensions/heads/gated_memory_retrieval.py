"""N2 -- GatedMemoryRetrieval: keeps CB_TA's LSTM exactly as-is and ADDS a
zero-init-gated retrieval read over a compressed bank of the piece's own
past conditioning vectors.

RELATION TO N1. N1 replaces the LSTM with a memory Transformer (bigger
change, must learn temporal modelling from scratch). N2 targets the same
measured failure -- burst mislocalisation on repeat-heavy pieces, where the
median onset error is 0.000s but the mean is 1.3-12.4s -- with the
conservative variant that this project's own results say is the reliable
pattern: leave the converged network untouched and bolt on a gated,
additive path.

That pattern is not a guess. Every FiLM REPLACEMENT tried here lost ground
against plain FiLM (spatial 44.3%, cross-attention 71.1%, gated 82.9%, vs
89.2% for B1a), while the one change that actually beat B1a was B3 (89.8%),
an ADDITIVE auxiliary trained on top of a converged B1a checkpoint. So here
the LSTM is not merely kept -- it is kept as `network.rnn` with its
parameter names unchanged (`rnn.weight_ih_l0`, ...), so a warm start from
B1a's checkpoint restores it bit-for-bit, and the zero-initialised gate
makes the network's output at step zero EXACTLY B1a's output. The retrieval
path can then only earn its way in.

WHY A LAG EMBEDDING IS THE POINT. Retrieving "audio that resembles now" is
by itself ambiguous: on a repeat, the best match is the first time through,
and naively mixing in that memory argues for the position the model must NOT
choose. What disambiguates a repeat is not the match but its AGE -- "this
matches something ~40s ago, therefore I am on the second pass, therefore the
correct position is the later one." So every memory slot's key and value are
conditioned on an explicit sinusoidal encoding of how long ago it was, and
the network is free to learn that a strong old match implies advance rather
than return. Without the lag term this module could not represent the fix
the failure analysis calls for.

COMPRESSED BANK. Repeats in the failing pieces are tens of seconds apart, so
the bank stores average-pooled summaries: `n_mem` slots of `pool` frames
each, i.e. n_mem*pool frames of coverage (defaults: 192 x 16 = 3072 frames
~= 2.5 min at 20fps).

STATE PLUMBING. Packed into CPJKU's existing 2-tuple `hidden` so
iterate_dataset and eval_model.py are untouched (see
extensions/heads/long_context_temporal.py's docstring for the full argument
that this is safe -- iterate_dataset only ever zeroes/slices dim 1):
    hidden[0][0:1]   LSTM h            hidden[1][0:1]   LSTM c
    hidden[0][1:]    bank contents     hidden[1][1:][:, :, 0] validity flag
The patched network reports `rnn_layers = 1 + n_mem`.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lag_encoding(n_slots: int, dim: int, device) -> torch.Tensor:
    """(n_slots, dim) sinusoidal encoding of AGE. Row i is the slot i
    positions from the oldest end, so age decreases with i; the most recent
    slot (last row) gets age 0."""
    age = torch.arange(n_slots - 1, -1, -1, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(n_slots, dim, device=device)
    pe[:, 0::2] = torch.sin(age * div)
    pe[:, 1::2] = torch.cos(age * div)
    return pe


class GatedMemoryRetrieval(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, n_mem: int = 192, pool: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_mem = n_mem
        self.pool = pool

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        # Keys and values both see content + how long ago it was.
        self.k_proj = nn.Linear(2 * d_model, d_model)
        self.v_proj = nn.Linear(2 * d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)

        # The branch is silenced at init by ZERO-INITIALISING ITS LAST LAYER
        # (out_proj), not by a zero multiplicative gate. Both give exact
        # identity, but a zero gate also zeroes the gradient of everything
        # inside the branch (d/dtheta of gate*f(x) is gate*df/dtheta = 0), so
        # the branch cannot begin learning -- and unlike N3's, this branch's
        # output at init is RANDOM (attention over randomly projected memory),
        # so the gate would have no informative direction to grow in either.
        # It would random-walk near zero and the retrieval path could never
        # switch on. Zero-initialising out_proj instead leaves
        # d/d(out_proj.W) = dL/dout * attn_out != 0, so out_proj moves on step
        # one and unblocks the rest of the branch immediately. This is the
        # standard zero-init-final-layer trick (GPT-2 residual scaling,
        # ControlNet's zero convolutions); scripts/smoke_test_temporal_arch.py
        # asserts both the identity and the unblocking.
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # ConditionalUNet.__init__ ends with self.apply(initialize_weights),
        # which orthogonal-inits EVERY nn.Linear it finds and would silently
        # undo the line above (the exact bug already caught once in this repo
        # for GatedFiLM). temporal_arch_patch wraps initialize_weights to
        # honour this tag.
        self.out_proj._zero_init = True
        # Learnable per-channel scale (LayerScale-style), starting at 1 so it
        # is a scale and not a second gradient blocker.
        self.gate = nn.Parameter(torch.ones(1, 1, d_model))

    @property
    def state_depth(self) -> int:
        return 1 + self.n_mem

    def split_state(self, hidden):
        """-> ((h, c), bank (n_mem, bs, D), valid (n_mem, bs) bool)"""
        a, b = hidden
        lstm_hidden = (a[0:1].contiguous(), b[0:1].contiguous())
        bank = a[1:]
        valid = b[1:, :, 0] > 0.5
        return lstm_hidden, bank, valid

    def pack_state(self, lstm_hidden, bank, valid):
        h, c = lstm_hidden
        a = torch.cat([h, bank], dim=0)
        flag = torch.zeros_like(bank)
        flag[:, :, 0] = valid.to(bank.dtype)
        b = torch.cat([c, flag], dim=0)
        return a, b

    def read(self, rnn_out, bank, valid):
        """rnn_out: (L, bs, D) the LSTM's output (CB_TA's FiLM vector).
        Returns the same shape, refined by the gated retrieval path."""
        L, bs, D = rnn_out.shape
        n = bank.shape[0]
        if n == 0:
            return rnn_out
        lag = _lag_encoding(n, D, rnn_out.device).unsqueeze(1).expand(n, bs, D)
        kv_in = self.norm_kv(bank)
        kv_in = torch.cat([kv_in, lag], dim=-1)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        q = self.norm_q(rnn_out)
        # A slot is ignorable only if it was never written. Guard the
        # all-invalid case (start of a piece): unmask slot 0 so softmax never
        # sees an all -inf row, then rely on the gate//zeroed content.
        kpm = ~valid.transpose(0, 1)                       # (bs, n) True = ignore
        all_masked = kpm.all(dim=1)
        if all_masked.any():
            kpm = kpm.clone()
            kpm[all_masked, 0] = False
        attn_out, _ = self.attn(q, k, v, key_padding_mask=kpm, need_weights=False)
        return rnn_out + self.gate * self.out_proj(attn_out)

    def update(self, bank, valid, rnn_out):
        """Append pooled summaries of this chunk's conditioning vectors."""
        L, bs, D = rnn_out.shape
        n_slots = max(1, math.ceil(L / self.pool))
        pad = n_slots * self.pool - L
        x = F.pad(rnn_out.detach().permute(1, 2, 0), (0, pad), mode='replicate')
        pooled = F.avg_pool1d(x, kernel_size=self.pool, stride=self.pool).permute(2, 0, 1)
        new_bank = torch.cat([bank, pooled], dim=0)[-self.n_mem:]
        new_valid = torch.cat(
            [valid, torch.ones(n_slots, bs, dtype=torch.bool, device=rnn_out.device)],
            dim=0)[-self.n_mem:]
        return new_bank, new_valid
