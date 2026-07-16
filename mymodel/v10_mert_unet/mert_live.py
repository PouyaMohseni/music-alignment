"""Live, fine-tunable MERT-v1-95M encoder for end-to-end training.

Every other MERT-based experiment in this project (v13/v14/v15's
MERTProjector, B1a's frozen swap) consumes PRECOMPUTED, FROZEN MERT
embeddings (scripts/precompute_mert_zenodo.py) -- MERT itself has never
been part of any training graph in this project. Its self-supervised
pretraining objective was never optimized for this task's 50ms-precision
temporal localization, and a frozen readout can't adapt that. This module
runs the real HF MERT-v1-95M model live, on a local raw-audio window per
BPTT chunk, with gradients enabled, so (at least some of) its weights can
actually adapt.

MERTModel (m-a-p/MERT-v1-95M, trust_remote_code=True) subclasses HF's
HubertModel: `.feature_extractor` is the conv front-end (frozen here,
standard wav2vec2-style practice), `.encoder.layers` is a plain
nn.ModuleList of transformer blocks (confirmed by reading the cached
modeling_MERT.py directly -- MERTModel(HubertModel), self.encoder is one of
HubertEncoder/HubertEncoderStableLayerNorm/HubertEncoder_extend, all of
which expose `.layers`).

Resampling from MERT's native 75fps output to this project's 20fps
convention is reimplemented with torch.nn.functional.interpolate (NOT
scipy.interpolate, like scripts/precompute_mert_zenodo.py uses -- that
detaches gradients) so gradients flow all the way back into MERT's
transformer weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

MERT_SR = 24000
MERT_FPS = 75


class MERTLive(nn.Module):
    def __init__(self, mert_id: str = 'm-a-p/MERT-v1-95M', unfreeze_last_n: int | None = None):
        """unfreeze_last_n: None => fine-tune the WHOLE encoder stack (every
        transformer layer trainable). An int N freezes every layer except the
        top N transformer blocks -- cheaper, less prone to catastrophic
        forgetting of the pretrained representation. The conv feature
        extractor front-end is always frozen either way (standard wav2vec2-
        style fine-tuning practice -- it's a low-level signal-processing
        front-end, not where task adaptation should happen)."""
        super().__init__()
        self.model = AutoModel.from_pretrained(mert_id, trust_remote_code=True)

        for p in self.model.parameters():
            p.requires_grad_(False)
        if hasattr(self.model, 'feature_extractor'):
            for p in self.model.feature_extractor.parameters():
                p.requires_grad_(False)

        encoder_layers = self.model.encoder.layers
        n_total = len(encoder_layers)
        unfreeze_from = 0 if unfreeze_last_n is None else max(0, n_total - unfreeze_last_n)
        n_trainable = 0
        for i, layer in enumerate(encoder_layers):
            if i >= unfreeze_from:
                for p in layer.parameters():
                    p.requires_grad_(True)
                    n_trainable += p.numel()
        print(f'[MERTLive] {n_total} encoder layers, trainable from layer {unfreeze_from} '
              f'({n_trainable:,} trainable params in the encoder stack)', flush=True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def embed_window(self, wav_24k: torch.Tensor, n_frames_20fps: int) -> torch.Tensor:
        """wav_24k: (n_samples,) float32 raw audio @ 24kHz, on the target
        device, already sliced to the window the caller wants MERT to
        attend over (this fn does not do any windowing itself). Returns
        (n_frames_20fps, 768) -- the LAST n_frames_20fps frames of this
        window's output, resampled to 20fps. Callers should include enough
        left-context in wav_24k before the frames they actually want, since
        MERT's self-attention needs surrounding context, not just the exact
        samples for the target frames."""
        out = self.model(input_values=wav_24k.unsqueeze(0)).last_hidden_state[0]  # (T75, 768)
        t75 = out.shape[0]
        out_bct = out.transpose(0, 1).unsqueeze(0)  # (1, 768, T75)
        t20 = max(1, round(t75 * 20 / MERT_FPS))
        resampled = F.interpolate(out_bct, size=t20, mode='linear', align_corners=True)
        resampled = resampled.squeeze(0).transpose(0, 1)  # (T20, 768)
        if resampled.shape[0] < n_frames_20fps:
            pad = n_frames_20fps - resampled.shape[0]
            resampled = F.pad(resampled, (0, 0, pad, 0))
        return resampled[-n_frames_20fps:]
