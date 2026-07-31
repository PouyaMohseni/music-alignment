"""N3 -- GatedBeliefPropagation: a differentiable Bayesian filter over the
score position, fused into CB_TA as a zero-init-gated additive log-prior on
the output heatmap.

THE FAILURE THIS IS BUILT FOR. In B1a (89.2% pct@0.5s) the failing pieces do
not degrade gracefully: their MEDIAN onset error is 0.000s while the MEAN is
1.3-12.4s. The model is exact most of the time and then teleports -- it
jumps to a visually similar passage, dwells, and recovers. CB_TA predicts
each frame's heatmap independently given the LSTM's conditioning vector;
nothing in the architecture forbids the predicted position from moving
across the page between consecutive frames. A real score follower's position
cannot teleport, and that constraint is simply absent from the model.

WHAT THIS ADDS. A proper recursive Bayes filter over a coarse 2D belief
b_t(y, x) about where in the page we are:

    prior_t    = (1 - j) * (b_{t-1} * K)  +  j * uniform        [predict]
    b_t        = softmax( log prior_t + evidence_t )            [update]
    logit bias = gate * log prior_t                             [inject]

K is a learned 2D transition kernel (initialised to favour a small rightward
step, i.e. reading along a staff, but free to learn the down-and-left jump
that a system wrap requires -- which is why the kernel is 2D and general
rather than a monotone forward shift; on a multi-system page x is NOT
monotone in time). `evidence_t` is the network's own per-frame heatmap,
coarse-pooled -- so the CNN supplies the likelihood and this layer supplies
the temporal prior, which is the correct factorisation rather than a
post-hoc smoother.

THE UNIFORM FLOOR IS LOad-BEARING. j = sigmoid(learned scalar) mixes a
uniform distribution into every prediction step. Without it a confident
filter assigns near-zero prior to any distant position, and a single wrong
commitment becomes unrecoverable -- the filter would CAUSE exactly the
dwelling failure it is meant to cure. The floor guarantees the posterior can
always escape, making this a robust tracker rather than a lock-in device.
This is the standard transition-noise floor from HMM tracking.

WHY GATED AND ADDITIVE. Every FiLM REPLACEMENT tried in this project lost
ground (spatial 44.3%, cross-attention 71.1%, gated 82.9%, vs plain FiLM
89.2%), while the only change that beat B1a was B3 (89.8%), an additive
auxiliary on top of a converged checkpoint. So the gate is zero-initialised:
at step zero the bias term vanishes and the network's output is EXACTLY
B1a's, and the filter has to earn its influence.

STATE PLUMBING. The belief is carried in CPJKU's existing 2-tuple `hidden`
(see extensions/heads/long_context_temporal.py's docstring for why this is
safe: iterate_dataset only zeroes and slices dim 1, which is exactly the
per-piece reset and slot-drop semantics the belief needs):
    hidden[0][0:1]  LSTM h        hidden[1][0:1]  LSTM c
    hidden[0][1:1+n_rows]  belief, flattened over (Hb, Wb)
    hidden[1][1:2][:, :, 0] validity flag
The patched network reports `rnn_layers = 1 + n_rows`.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


class GatedBeliefPropagation(nn.Module):
    def __init__(self, d_model: int = 512, belief_h: int = 16, belief_w: int = 64,
                 kernel_h: int = 5, kernel_w: int = 9):
        super().__init__()
        self.d_model = d_model
        self.bh = belief_h
        self.bw = belief_w
        self.kh = kernel_h
        self.kw = kernel_w
        self.n_cells = belief_h * belief_w
        self.n_rows = int(math.ceil(self.n_cells / d_model))

        # Transition kernel as unnormalised logits; softmax over all taps at
        # use time makes it a genuine probability kernel (mass-preserving).
        k = torch.zeros(kernel_h, kernel_w)
        cy, cx = kernel_h // 2, kernel_w // 2
        for dy in range(kernel_h):
            for dx in range(kernel_w):
                # Favour staying put / stepping slightly right; mild, since
                # the gate is zero at init and this only shapes early learning.
                k[dy, dx] = -0.5 * abs(dy - cy) - 0.35 * abs(dx - (cx + 1))
        self.kernel_logits = nn.Parameter(k)
        # Uniform-mixture (transition noise floor) logit; sigmoid(-2.2) ~= 0.1.
        self.jump_logit = nn.Parameter(torch.tensor(-2.2))
        # Zero-init gate -- a bare Parameter, so ConditionalUNet.__init__'s
        # trailing self.apply(initialize_weights) cannot clobber it.
        self.gate = nn.Parameter(torch.zeros(1))
        # Learned scaling of the CNN evidence entering the filter.
        self.evidence_scale = nn.Parameter(torch.ones(1))

    @property
    def state_depth(self) -> int:
        return 1 + self.n_rows

    def split_state(self, hidden):
        a, b = hidden
        lstm_hidden = (a[0:1].contiguous(), b[0:1].contiguous())
        bs = a.shape[1]
        flat = a[1:1 + self.n_rows].permute(1, 0, 2).reshape(bs, -1)[:, :self.n_cells]
        valid = b[1:2, :, 0] > 0.5                      # (1, bs)
        belief = flat.view(bs, self.bh, self.bw)
        return lstm_hidden, belief, valid.squeeze(0)

    def pack_state(self, lstm_hidden, belief, valid):
        h, c = lstm_hidden
        bs = belief.shape[0]
        flat = belief.reshape(bs, -1)
        pad = self.n_rows * self.d_model - flat.shape[1]
        if pad > 0:
            flat = F.pad(flat, (0, pad))
        rows = flat.view(bs, self.n_rows, self.d_model).permute(1, 0, 2)
        a = torch.cat([h, rows], dim=0)
        flag = torch.zeros_like(rows)
        flag[0, :, 0] = valid.to(rows.dtype)
        b = torch.cat([c, flag], dim=0)
        return a, b

    def _transition(self, belief):
        """belief (bs, Hb, Wb) -> prior (bs, Hb, Wb), mass-preserving."""
        k = torch.softmax(self.kernel_logits.view(-1), dim=0).view(1, 1, self.kh, self.kw)
        x = belief.unsqueeze(1)
        x = F.pad(x, (self.kw // 2, self.kw // 2, self.kh // 2, self.kh // 2), mode='replicate')
        moved = F.conv2d(x, k).squeeze(1)
        j = torch.sigmoid(self.jump_logit)
        prior = (1.0 - j) * moved + j / float(self.n_cells)
        return prior / prior.sum(dim=(1, 2), keepdim=True).clamp_min(_EPS)

    def forward(self, logits, hidden_belief, valid, seq_len, bs):
        """logits: (seq_len*bs, 1, H, W) pre-sigmoid CNN output.
        Returns (biased_logits, new_belief, new_valid)."""
        H, W = logits.shape[-2], logits.shape[-1]
        # Coarse log-domain evidence. Max-pool is the log-domain 'any cell in
        # this region is active', matching the heatmap's peaked structure.
        ev = F.adaptive_max_pool2d(logits, (self.bh, self.bw))       # (SB,1,bh,bw)
        ev = ev.view(seq_len, bs, self.bh, self.bw) * self.evidence_scale

        belief = hidden_belief
        # A never-written slot starts uniform.
        uniform = torch.full_like(belief, 1.0 / self.n_cells)
        belief = torch.where(valid.view(bs, 1, 1), belief, uniform)
        belief = belief.clamp_min(0)
        belief = belief / belief.sum(dim=(1, 2), keepdim=True).clamp_min(_EPS)

        biases = []
        for t in range(seq_len):
            prior = self._transition(belief)
            log_prior = torch.log(prior + _EPS)
            biases.append(log_prior)
            post = torch.softmax((log_prior + ev[t]).view(bs, -1), dim=-1)
            belief = post.view(bs, self.bh, self.bw)

        bias = torch.stack(biases, dim=0).view(seq_len * bs, 1, self.bh, self.bw)
        # Centre the bias so the gate injects SHAPE, not a global offset that
        # would just rescale every logit and fight the dice loss.
        bias = bias - bias.mean(dim=(2, 3), keepdim=True)
        bias_up = F.interpolate(bias, size=(H, W), mode='bilinear', align_corners=False)
        out = logits + self.gate * bias_up

        new_valid = torch.ones(bs, dtype=torch.bool, device=logits.device)
        return out, belief, new_valid
