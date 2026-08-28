"""A small listwise selector over the frozen detector's candidate boxes.

It replaces the four hand-tuned constants of the transition prior (fwd, sigma,
jump, lam) with a learned scoring function, and it is deliberately tiny: about
10k parameters over 20 piece-agnostic features. The unit of data is a
(frame, candidate) pair, so the training set is tens of thousands of examples
rather than the 353 pieces that sank every previous attempt to add capacity.

PERMUTATION EQUIVARIANCE
------------------------
Scoring each candidate in isolation cannot express "this one is the clear
leader" versus "three are tied", which is exactly the situation the hand-tuned
prior handles badly. So each candidate is embedded, the frame is summarised by
mean- and max-pooling those embeddings, and the score is read off from the
candidate together with its frame context. The detector emits candidates in no
meaningful order beyond objectness, so the architecture is equivariant by
construction and cannot learn anything from position in the list.

Feature normalisation is stored IN the module as buffers, so a checkpoint
carries its own preprocessing and inference cannot silently use different
statistics from training.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from extensions.heads.cand_features import NF


class CandScorer(nn.Module):
    def __init__(self, nf: int = NF, hidden: int = 64, embed: int = 32,
                 use_abs_obj: bool = True, zdim: int = 0, zproj: int = 16):
        super().__init__()
        self.nf, self.use_abs_obj = nf, use_abs_obj
        self.zdim, self.zproj = zdim, zproj if zdim else 0
        self.enc = nn.Sequential(
            nn.Linear(nf, hidden), nn.ReLU(),
            nn.Linear(hidden, embed), nn.ReLU())
        # z is the detector's own audio representation, shared by every
        # candidate in a frame, so it belongs in the FRAME CONTEXT rather than
        # per candidate -- it says what the music is doing, not which box is
        # right. It enters through a small projection (128 -> 16) because the
        # training set is 353 pieces and a full-width 128-dim path into a 10k
        # model is the shape of every capacity-add that has failed here.
        self.zenc = (nn.Sequential(nn.Linear(zdim, zproj), nn.ReLU())
                     if zdim else None)
        self.head = nn.Sequential(
            nn.Linear(embed * 3 + self.zproj, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        if zdim:
            self.register_buffer('zmu', torch.zeros(zdim))
            self.register_buffer('zsd', torch.ones(zdim))
        self.register_buffer('mu', torch.zeros(nf))
        self.register_buffer('sd', torch.ones(nf))

    def set_norm(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-3))

    def set_znorm(self, mu, sd):
        if self.zenc is None:
            return
        self.zmu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.zsd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-3))

    def forward(self, x, mask=None, z=None):
        """x: (B, K, nf). mask: (B, K) bool, True where the slot is a real
        candidate. Returns (B, K) scores with padded slots at -inf."""
        h = self.enc((x - self.mu) / self.sd)
        if mask is None:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        m = mask.unsqueeze(-1)
        n = m.sum(1).clamp_min(1)
        mean = (h * m).sum(1) / n
        mx = h.masked_fill(~m, float('-inf')).max(1).values
        mx = torch.nan_to_num(mx, neginf=0.0)          # a frame with no candidates
        parts = [mean, mx]
        if self.zenc is not None:
            if z is None:
                z = torch.zeros(h.shape[0], self.zdim, device=h.device)
            parts.append(self.zenc((z - self.zmu) / self.zsd))
        ctx = torch.cat(parts, -1).unsqueeze(1).expand(-1, h.shape[1], -1)
        s = self.head(torch.cat([h, ctx], -1)).squeeze(-1)
        return s.masked_fill(~mask, float('-inf'))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def save(model, path, extra=None):
    torch.save({'state_dict': model.state_dict(), 'nf': model.nf,
                'hidden': model.enc[0].out_features,
                'embed': model.enc[2].out_features,
                'use_abs_obj': model.use_abs_obj, 'zdim': model.zdim,
                'zproj': model.zproj or 16, 'extra': extra or {}}, path)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = CandScorer(nf=ck['nf'], hidden=ck['hidden'], embed=ck['embed'],
                   use_abs_obj=ck['use_abs_obj'], zdim=ck.get('zdim', 0),
                   zproj=ck.get('zproj', 16))
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m, ck.get('extra', {})
