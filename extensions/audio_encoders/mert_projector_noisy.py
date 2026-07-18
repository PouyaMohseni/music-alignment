"""MERTProjectorNoisy: approximates B6's spirit (impulse-response audio
augmentation at a controlled SNR, extensions/hooks/ir_patch.py) WITHOUT a
true acoustic-domain reproduction -- MERT embeddings are precomputed and
frozen, so there's no live audio-to-feature step left to convolve a
synthetic room response into. A faithful reproduction would mean rendering
IR-augmented audio copies and running MERT over each (a new multi-hour
precompute job), not attempted here.

Instead: inject Gaussian noise directly onto the raw 768-dim MERT embedding
at a controlled SNR (same snr_range_db parameterization as B6, applied to
the embedding vector's own norm rather than the waveform's), during
training only (respects self.training, same convention as e.g. Dropout).
This is a genuinely different, weaker form of robustness training than true
acoustic augmentation -- reported as an approximation, not a substitute.
"""
import torch
import torch.nn as nn


class MERTProjectorNoisy(nn.Module):
    def __init__(self, spec_enc, d_mert=768, p=0.5, snr_range_db=(10.0, 30.0)):
        super().__init__()
        self.n_input_frames = 1
        self.d_mert = d_mert
        self.p = p
        self.snr_lo_db, self.snr_hi_db = snr_range_db
        self.enc = nn.Sequential(
            nn.Linear(d_mert, 256),
            nn.ELU(),
            nn.Linear(256, spec_enc),
        )
        self.means = nn.Parameter(torch.zeros(d_mert, 1), requires_grad=False)
        self.stds  = nn.Parameter(torch.ones(d_mert, 1), requires_grad=False)

    def set_stats(self, means, stds):
        self.means = nn.Parameter(torch.from_numpy(means).view(self.d_mert, 1), requires_grad=False)
        self.stds  = nn.Parameter(torch.from_numpy(stds).view(self.d_mert, 1), requires_grad=False)

    def reshape_input(self, x):
        seq_len, bs, c, h, w = x.shape
        return x.view(seq_len * bs, -1)

    def _add_snr_noise(self, x: torch.Tensor) -> torch.Tensor:
        """x: (SB, 768), already normalized. Adds Gaussian noise scaled so
        the per-row signal-to-noise ratio matches a value drawn uniformly
        (in dB) from [snr_lo_db, snr_hi_db], only on rows selected by a
        Bernoulli(p) draw (matching B6's per-call augmentation probability)."""
        if not self.training or self.p <= 0:
            return x
        sb = x.shape[0]
        apply_mask = (torch.rand(sb, device=x.device) < self.p)
        if not apply_mask.any():
            return x
        snr_db = torch.empty(sb, device=x.device).uniform_(self.snr_lo_db, self.snr_hi_db)
        signal_power = x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8)   # (SB, 1)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0)).unsqueeze(1)
        noise = torch.randn_like(x) * noise_power.sqrt()
        noisy = x + noise
        return torch.where(apply_mask.unsqueeze(1), noisy, x)

    def forward(self, x):
        x = self.reshape_input(x)
        x = (x - self.means.view(1, -1)) / self.stds.view(1, -1)
        x = self._add_snr_noise(x)
        return self.enc(x)
