"""C1 -- Audio-visual grounding via cross-attention.

Replaces CB_TA's FiLM-modulated-CNN + LSTM decoder with a Transformer that
cross-attends from a per-frame audio query onto score-image patch tokens.
Audio and score-image inputs only -- no symbolic MIDI/MusicXML, matching
the project's actual inference constraint (audio + rendered score image).

Score side: the strip is tokenized into 1-D patches spanning the FULL
strip height in one patch (kernel=(H, patch_w), stride=(1, patch_w)) --
alignment only concerns x-position, and the existing GT masks are already
full-height vertical bars (see v11's make_gt_mask), so modeling a 2-D grid
of patches would spend capacity on a vertical dimension nothing needs.
This also keeps patch count in the tens (H=128, patch_w=32 -> ~W_sc/32
patches, e.g. ~23 for a 742px-wide scaled strip) rather than the hundreds+
a full 2-D ViT tokenization would give.

Audio side: reuses CBEncoder unchanged (same 40-frame mel window CB_TA/A0/
B1-B6 all use) followed by an LSTM for temporal context, matching the
existing training pipeline's per-frame windowing (see data.py's
_build_perf-equivalent in train.py) -- this keeps the same BPTT chunking
structure the training loop already uses.

Cross-attention: one query per audio frame attends over all score patch
tokens (shared keys/values -- the score doesn't change within a piece).
The attention weights (not the attended VALUE) are what we actually want:
they are directly a soft position distribution over the strip, so they are
reshaped to the 1-D patch grid and linearly upsampled back to the full
(H, W_sc) resolution to produce a heatmap compatible with the existing
dice-loss/center-of-mass decode infrastructure used everywhere else in
this project (extensions/hooks/position_decoder.py).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from mymodel.v9_cpjku.cpjku_audio import CBEncoder


class ScorePatchEncoder(nn.Module):
    def __init__(self, d_model: int, patch_w: int = 32):
        super().__init__()
        self.patch_w = patch_w
        self.d_model = d_model
        # kernel height is set dynamically to the strip height on first call
        # (H is fixed per config, but building the conv lazily avoids baking
        # H into __init__ and keeps this reusable if h_strip ever changes).
        self.patch_conv = None

    def _build(self, H: int, device):
        self.patch_conv = nn.Conv2d(1, self.d_model, kernel_size=(H, self.patch_w),
                                    stride=(1, self.patch_w), padding=(0, 0)).to(device)

    @staticmethod
    def _sinusoidal_pe(n: int, d_model: int, device) -> torch.Tensor:
        position = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(n, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        """score: (1, 1, H, W_sc) -> (num_patches, d_model), plus records
        num_patches and the padded width used (for the caller to upsample
        attention weights back to the ORIGINAL W_sc, not the padded one)."""
        H, W_sc = score.shape[-2], score.shape[-1]
        if self.patch_conv is None:
            self._build(H, score.device)

        pad_w = (self.patch_w - W_sc % self.patch_w) % self.patch_w
        if pad_w:
            score = F.pad(score, (0, pad_w))

        patches = self.patch_conv(score)          # (1, d_model, 1, num_patches)
        patches = patches.squeeze(2).squeeze(0).transpose(0, 1)   # (num_patches, d_model)
        pe = self._sinusoidal_pe(patches.shape[0], self.d_model, patches.device)
        return patches + pe, W_sc   # (num_patches, d_model), original (unpadded) width


class C1VisualGroundingNet(nn.Module):
    """Same forward(score, perf, hidden) interface as CPJKU's ConditionalUNet
    (see third_party/cpjku_unet/audio_conditioned_unet/network.py) so this
    plugs into the same BPTT training-loop shape conventions as v11 --
    but this is NOT that network; it needs its own train.py (see train.py
    in this directory), since the internal architecture is unrelated."""

    def __init__(self, spec_enc: int = 32, rnn_size: int = 128, rnn_layers: int = 1,
                d_model: int = 128, n_heads: int = 4, patch_w: int = 32):
        super().__init__()
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.use_lstm = True   # always -- no non-recurrent path for this model

        self.perf_encoder = CBEncoder(spec_enc)
        self.rnn = nn.LSTM(spec_enc, hidden_size=rnn_size, num_layers=rnn_layers, batch_first=False)
        self.audio_proj = nn.Linear(rnn_size, d_model)

        self.score_encoder = ScorePatchEncoder(d_model, patch_w=patch_w)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.first_execution = True

    def forward(self, score: torch.Tensor, perf: torch.Tensor, hidden):
        """score: (seq_len, bs=1, 1, H, W_sc) -- same value repeated across
              seq_len (the caller expands a single-frame tile, matching v11).
        perf:  (seq_len, bs=1, 1, n_mels, n_frames)
        hidden: LSTM (h, c) state or None.
        Returns: {'segmentation': (seq_len, 1, H, W_sc), 'hidden': (h, c)}
        """
        seq_len, bs, c, H, W_sc = score.shape
        assert bs == 1, 'C1 only supports bs=1 (variable strip widths, like v11)'

        # Score is identical across seq_len within one BPTT chunk -- only
        # patch-encode ONE frame, not seq_len copies of the same image.
        patch_tokens, orig_w = self.score_encoder(score[0])   # (num_patches, d_model)
        num_patches = patch_tokens.shape[0]

        audio_feat = self.perf_encoder(perf)             # (seq_len*bs, spec_enc)
        audio_feat = audio_feat.view(seq_len, bs, -1)
        audio_feat, hidden = self.rnn(audio_feat, hidden)  # (seq_len, bs, rnn_size)
        audio_feat = audio_feat.reshape(seq_len * bs, -1)
        query = self.audio_proj(audio_feat).unsqueeze(1)   # (seq_len, 1, d_model)

        kv = patch_tokens.unsqueeze(0).expand(seq_len, -1, -1)   # (seq_len, num_patches, d_model)

        _, attn_weights = self.cross_attn(query, kv, kv, need_weights=True,
                                          average_attn_weights=True)
        # attn_weights: (seq_len, 1, num_patches) -- this IS the position distribution
        heat_1d = attn_weights.squeeze(1)   # (seq_len, num_patches)

        # Upsample patch-resolution attention to full strip width, then
        # broadcast uniformly across height (alignment is x-only; existing
        # GT masks are already full-height bars -- see module docstring).
        heat_1d = heat_1d.unsqueeze(1)                          # (seq_len, 1, num_patches)
        pad_w = num_patches * self.score_encoder.patch_w
        heat_full = F.interpolate(heat_1d, size=pad_w, mode='linear', align_corners=False)
        heat_full = heat_full[..., :orig_w]                     # crop off patch padding
        heat_full = heat_full.unsqueeze(2).expand(-1, -1, H, -1)  # (seq_len, 1, H, W_sc)

        if self.first_execution:
            print(f'[C1] score {score.shape} -> patches {patch_tokens.shape} '
                  f'-> heatmap {heat_full.shape}', flush=True)
            self.first_execution = False

        return {'segmentation': heat_full.contiguous(), 'hidden': hidden}
