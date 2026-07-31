"""N1 -- LongContextTemporalCore: a two-tier memory Transformer that replaces
CB_TA's 1-layer LSTM as the temporal context module.

WHY THIS, AND WHY NOW (measured, not assumed). In the best model to date
(B1a native, 89.2% pct@0.5s) the residual error is NOT a precision problem:
across the failing pieces the MEDIAN onset error is 0.000s while the MEAN is
1.3-12.4s (e.g. SchumannR op68-01: median 0.000 / mean 5.66; ChopinFF
Op28-9: median 3.35 / mean 12.39). A median of zero with a large mean is the
signature of a tracker that is exactly right most of the time and
catastrophically wrong in BURSTS -- it jumps to a visually similar passage
(a repeat), dwells there, then recovers. Meanwhile ~75% of the test pieces
score 98-100%. So the remaining headroom is concentrated in discrete global
mislocalisation events on structurally ambiguous pieces, which is a
long-range-context problem.

CB_TA's temporal module is a single-layer LSTM whose entire memory of the
performance so far is one fixed-size 512-d state vector, trained by BPTT
over short chunks. Deciding "have I already played this passage once?"
requires comparing the current audio against the piece's own distant past --
precisely what a fixed-size recurrent bottleneck cannot represent and what
attention over an explicit history can.

TWO-TIER MEMORY (Compressive-Transformer style, Rae et al. 2019). Dense
attention over a whole piece (minutes; 20fps -> thousands of frames) is not
affordable, and a short window would not reach a repeat. So the history is
kept at two resolutions:
  - FINE tier: the last `n_fine` frames verbatim (local continuity).
  - COMPRESSED tier: `n_comp` slots, each an average-pool of `pool` frames,
    giving n_comp*pool frames of coverage for n_comp slots.
With the defaults below (64 fine, 192 compressed, pool 16) the model sees
3.2s at full rate plus ~2.5 minutes of coarse history at 20fps -- enough to
span the repeats that cause the failures above.

STATE PLUMBING -- no changes to iterate_dataset or eval_model.py. CPJKU's
iterate_dataset owns the recurrent state: it allocates
`torch.zeros(network.rnn_layers, batch_size, network.rnn_size)` twice, calls
`network(score=, perf=, hidden=)`, detaches between chunks, zeroes
`hidden[i][:, idx]` to reset one batch slot at a piece boundary, and
concatenates around dim 1 to drop a finished slot. Every one of those
operations is shape-agnostic on dim 0 and slot-wise on dim 1, so this module
simply PACKS its own memory into that same 2-tuple:
    hidden[0] : (n_fine + n_comp, bs, d_model)  the memory contents
    hidden[1] : (n_fine + n_comp, bs, d_model)  [:, :, 0] is the validity flag
and the patched network reports `rnn_layers = n_fine + n_comp`. A zeroed slot
is therefore an empty, all-invalid memory -- exactly the reset semantics
required -- and popping a slot slices the memory correctly for free. The
module also mimics nn.LSTM's call signature `(x, hidden) -> (out, hidden)`
and is installed as `network.rnn`, which is what iterate_dataset probes via
`hasattr(network, "rnn")`.

NaN SAFETY. Only the current chunk's frames are used as attention QUERIES;
memory is keys/values only. Every query can always attend to at least its own
position, so no query ever has a fully-masked key set (the standard way a
padded-memory transformer produces NaNs -- softmax over all -inf).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe(length: int, dim: int, device, offset: int = 0) -> torch.Tensor:
    """(length, dim) sinusoidal positional encoding starting at `offset`."""
    pos = torch.arange(offset, offset + length, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _MemoryAttentionLayer(nn.Module):
    """Pre-norm block: current-chunk queries attend over [memory || current],
    then a position-wise FFN. Memory is read-only within a layer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, cur, mem, attn_mask, key_padding_mask):
        """cur: (L, bs, D) current chunk. mem: (S, bs, D) memory (may be S=0)."""
        q = self.norm_q(cur)
        kv = self.norm_kv(torch.cat([mem, cur], dim=0))
        attn_out, _ = self.attn(q, kv, kv, attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask, need_weights=False)
        cur = cur + attn_out
        cur = cur + self.ff(self.norm_ff(cur))
        return cur


class LongContextTemporalCore(nn.Module):
    """Drop-in replacement for CB_TA's `nn.LSTM(spec_enc, rnn_size)`.

    Call signature matches nn.LSTM: forward(x, hidden) -> (out, hidden) with
    x (L, bs, d_in) and out (L, bs, d_model)."""

    def __init__(self, d_in: int, d_model: int, n_layers: int = 2, n_heads: int = 8,
                 n_fine: int = 64, n_comp: int = 192, pool: int = 16, dropout: float = 0.0):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_fine = n_fine
        self.n_comp = n_comp
        self.pool = pool

        self.in_proj = nn.Linear(d_in, d_model)
        self.tier_emb = nn.Embedding(3, d_model)   # 0 = compressed, 1 = fine, 2 = current
        self.layers = nn.ModuleList([
            _MemoryAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)

    @property
    def state_depth(self) -> int:
        """Depth iterate_dataset must allocate, i.e. the patched network's
        `rnn_layers`. Layout along dim 0: [compressed | fine]."""
        return self.n_comp + self.n_fine

    def _split_state(self, hidden):
        mem, flag = hidden
        valid = flag[:, :, 0] > 0.5                     # (S, bs)
        return mem, valid

    def forward(self, x, hidden):
        L, bs, _ = x.shape
        device = x.device
        h = self.in_proj(x)                              # (L, bs, D)

        if hidden is None:
            S = self.state_depth
            mem = torch.zeros(S, bs, self.d_model, device=device, dtype=h.dtype)
            valid = torch.zeros(S, bs, dtype=torch.bool, device=device)
        else:
            mem, valid = self._split_state(hidden)
            mem = mem.to(h.dtype)
        S = mem.shape[0]

        # --- positional / tier encoding -------------------------------------
        # Absolute index within [memory || current]; the window slides with the
        # piece, so this stays consistent step to step.
        pe = _sinusoidal_pe(S + L, self.d_model, device)                   # (S+L, D)
        tier = torch.empty(S + L, dtype=torch.long, device=device)
        tier[:self.n_comp] = 0
        tier[self.n_comp:S] = 1
        tier[S:] = 2
        enc = pe + self.tier_emb(tier)                                      # (S+L, D)

        mem_in = mem + enc[:S].unsqueeze(1)
        cur = h + enc[S:].unsqueeze(1)

        # --- masks ----------------------------------------------------------
        # Queries are ONLY the current chunk. Memory is fully visible (subject
        # to validity); within the chunk, strictly causal.
        attn_mask = torch.zeros(L, S + L, dtype=torch.bool, device=device)
        attn_mask[:, S:] = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        key_padding_mask = torch.zeros(bs, S + L, dtype=torch.bool, device=device)
        key_padding_mask[:, :S] = ~valid.transpose(0, 1)                    # True = ignore

        for layer in self.layers:
            cur = layer(cur, mem_in, attn_mask, key_padding_mask)
        out = self.norm_out(cur)                                            # (L, bs, D)

        # --- memory update ---------------------------------------------------
        # Buffers hold in_proj outputs (not layer outputs) so the memory is a
        # stable record of what was heard, independent of the evolving stack.
        fine_old = mem[self.n_comp:]
        fine_valid_old = valid[self.n_comp:]
        fine = torch.cat([fine_old, h.detach()], dim=0)[-self.n_fine:]
        fine_v = torch.cat([fine_valid_old,
                            torch.ones(L, bs, dtype=torch.bool, device=device)],
                           dim=0)[-self.n_fine:]

        n_slots = max(1, math.ceil(L / self.pool))
        pad = n_slots * self.pool - L
        h_pad = F.pad(h.detach().permute(1, 2, 0), (0, pad), mode='replicate')   # (bs, D, L+pad)
        pooled = F.avg_pool1d(h_pad, kernel_size=self.pool, stride=self.pool)     # (bs, D, n_slots)
        pooled = pooled.permute(2, 0, 1)                                          # (n_slots, bs, D)
        comp = torch.cat([mem[:self.n_comp], pooled], dim=0)[-self.n_comp:]
        comp_v = torch.cat([valid[:self.n_comp],
                            torch.ones(n_slots, bs, dtype=torch.bool, device=device)],
                           dim=0)[-self.n_comp:]

        new_mem = torch.cat([comp, fine], dim=0)
        new_valid = torch.cat([comp_v, fine_v], dim=0)
        new_flag = torch.zeros_like(new_mem)
        new_flag[:, :, 0] = new_valid.to(new_mem.dtype)
        return out, (new_mem, new_flag)
