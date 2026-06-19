"""v4 — Pitch-fused full-sequence alignment head.

Same lightweight head as v3 (operates on precomputed frozen MERT/ViT embeddings),
PLUS two auxiliary pitch heads that predict an 88-key pitch-roll on each side,
supervised by the MIDI ground truth at TRAIN time only. The matching similarity
is computed in a fused space:

    a = [ L2(proj_audio) , alpha * L2(sigmoid(audio_pitch_logits)) ]
    i = [ L2(proj_image) , alpha * L2(sigmoid(score_pitch_logits)) ]
    sim = a . i^T

So cosine similarity becomes pitch-aware. At inference the model still consumes
only (image, audio) — the pitch heads are an internal learned representation, not
a symbolic pivot. This directly attacks RC2 (pitch-blind features) while staying
end-to-end; E1 showed perfect pitch lifts mean error 5.35s -> 0.71s.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PITCH = 88


@dataclass
class PitchFusedConfig:
    d_audio: int = 768
    d_image: int = 768
    shared_dim: int = 256
    n_heads: int = 4
    n_cross_layers: int = 1
    dropout: float = 0.1
    pitch_fuse_alpha: float = 1.0    # weight of the pitch block in the fused vector
    pitch_hidden: int = 256


class _ProjHead(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.proj(self.drop(self.norm(x)))


class _PitchHead(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, N_PITCH))

    def forward(self, x):
        return self.net(x)              # logits (..., 88)


class _CrossAttn(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, q, ctx):
        a, _ = self.attn(q, ctx, ctx)
        x = self.norm(q + a)
        return self.norm2(x + self.ff(x))


class PitchFusedModel(nn.Module):
    """forward(audio_emb (B,T,Da), tile_emb (B,N,Di)) -> dict:
        sim                (B,T,N) fused cosine similarity
        audio_pitch_logits (B,T,88)
        score_pitch_logits (B,N,88)
    """

    def __init__(self, cfg: PitchFusedConfig | None = None):
        super().__init__()
        self.cfg = cfg or PitchFusedConfig()
        d = self.cfg.shared_dim
        self.audio_proj = _ProjHead(self.cfg.d_audio, d, self.cfg.dropout)
        self.image_proj = _ProjHead(self.cfg.d_image, d, self.cfg.dropout)
        self.audio_pitch = _PitchHead(self.cfg.d_audio, self.cfg.pitch_hidden)
        self.score_pitch = _PitchHead(self.cfg.d_image, self.cfg.pitch_hidden)
        self.audio_layers = nn.ModuleList(
            [_CrossAttn(d, self.cfg.n_heads, self.cfg.dropout) for _ in range(self.cfg.n_cross_layers)])
        self.image_layers = nn.ModuleList(
            [_CrossAttn(d, self.cfg.n_heads, self.cfg.dropout) for _ in range(self.cfg.n_cross_layers)])

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, audio_emb, tile_emb):
        a = self.audio_proj(audio_emb)
        i = self.image_proj(tile_emb)
        for la, li in zip(self.audio_layers, self.image_layers):
            a, i = la(a, i), li(i, a)

        a_pitch_logits = self.audio_pitch(audio_emb)     # (B,T,88)
        i_pitch_logits = self.score_pitch(tile_emb)      # (B,N,88)

        a_emb = F.normalize(a, dim=-1)
        i_emb = F.normalize(i, dim=-1)
        a_pit = F.normalize(torch.sigmoid(a_pitch_logits), dim=-1) * self.cfg.pitch_fuse_alpha
        i_pit = F.normalize(torch.sigmoid(i_pitch_logits), dim=-1) * self.cfg.pitch_fuse_alpha

        a_fused = torch.cat([a_emb, a_pit], dim=-1)
        i_fused = torch.cat([i_emb, i_pit], dim=-1)
        sim = torch.einsum("btd,bnd->btn", a_fused, i_fused)

        return {"sim": sim,
                "audio_pitch_logits": a_pitch_logits,
                "score_pitch_logits": i_pitch_logits}
