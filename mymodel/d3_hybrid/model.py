"""D3 -- hybrid of D1/D2's two-tower+DTW architecture with v13's proven,
trained audio representation, swapped in for D1/D2's own small conv-based
AudioTower. Tests whether D2's decode-quality win (D1: 17.2% -> D2: 55.0%
pct@0.5s from training-signal changes alone) compounds with a materially
stronger per-frame audio encoder.

Why v13's encoder, not v14's: v14's MERTBiLSTM (mymodel/v9_cpjku/cpjku_audio.py)
is fundamentally WINDOWED -- it consumes an 8-frame MERT window and pools to a
SINGLE vector via the BiLSTM's final hidden state (input (sl,bs,1,768,8) ->
output (sl*bs, spec_enc), one vector per window, no per-frame sequence output).
D1Model's similarity matrix needs a genuine (T, d) per-frame embedding
sequence over the WHOLE piece, not one call per fixed window -- forcing
v14's windowed design into that contract would mean re-deriving its exact
windowing/padding convention or risk a subtle mismatch with how it was
trained. v13's MERTProjector (same file) has n_input_frames=1 -- it is a pure
per-frame pointwise Linear+LayerNorm+ELU with NO temporal windowing at all,
so it extends to a whole (T, 768) sequence trivially and correctly. v13 is
also empirically the strongest of the three anyway (66.1% vs v14's 66.0% and
v15's 65.3%), so this loses nothing on the "strongest encoder" criterion.

Score tower is D1's, unchanged (never identified as the weak point).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mymodel.v9_cpjku.cpjku_audio import MERTProjector
from mymodel.d1_align_matrix.model import ScoreTower


class HybridAudioTower(nn.Module):
    """Wraps v13's trained MERTProjector (768 -> spec_enc, per-frame,
    no windowing) with a projection to D1Model's d_model and L2-normalization,
    matching AudioTower's exact output contract: (T, d_model) L2-normalized."""

    def __init__(self, d_mert: int = 768, spec_enc: int = 32, d_model: int = 128):
        super().__init__()
        self.mert_projector = MERTProjector(spec_enc=spec_enc, mert_dim=d_mert)
        self.out_proj = nn.Linear(spec_enc, d_model)

    def load_pretrained_v13(self, ckpt_path: str, device='cpu'):
        """Loads ONLY the perf_encoder.{proj,norm}.* weights from v13's
        trained CB_TA-style checkpoint into self.mert_projector -- verified
        exact shape/name match against mymodel.v9_cpjku.cpjku_audio.MERTProjector
        (proj.weight (spec_enc, d_mert), proj.bias, norm.weight, norm.bias;
        v13's checkpoint stores these under the 'perf_encoder.' prefix, its own
        ConditionalUNet's attribute name for the audio encoder)."""
        ck = torch.load(ckpt_path, map_location=device)
        sd = ck['state_dict'] if 'state_dict' in ck else ck
        prefix = 'perf_encoder.'
        sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = self.mert_projector.load_state_dict(sub_sd, strict=True)
        return missing, unexpected

    def forward(self, mert: torch.Tensor) -> torch.Tensor:
        """mert: (T, d_mert) -> (T, d_model) L2-normalized per frame."""
        T = mert.shape[0]
        x = mert.view(T, 1, 1, -1, 1)          # (sl=T, bs=1, c=1, d_mert, w=1) -- MERTProjector's expected shape
        x = self.mert_projector(x)              # (T, spec_enc)
        x = self.out_proj(x)                    # (T, d_model)
        return F.normalize(x, dim=-1)


class D3Model(nn.Module):
    def __init__(self, d_mert: int = 768, spec_enc: int = 32, d_model: int = 128,
                w_downsample: int = 4, n_ctx_layers: int = 2, n_heads: int = 4,
                temperature: float = 0.07):
        super().__init__()
        self.audio_tower = HybridAudioTower(d_mert, spec_enc, d_model)
        self.score_tower = ScoreTower(d_model, w_downsample, n_ctx_layers, n_heads)
        self.temperature = temperature
        self.w_downsample = w_downsample

    def encode(self, mert: torch.Tensor, strip: torch.Tensor):
        return self.audio_tower(mert), self.score_tower(strip)

    def similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (A @ B.t()) / self.temperature

    def forward(self, mert: torch.Tensor, strip: torch.Tensor) -> torch.Tensor:
        A, B = self.encode(mert, strip)
        return self.similarity(A, B)
