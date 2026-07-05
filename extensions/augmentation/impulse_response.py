"""B6 -- Real-Audio Robustness via Impulse Response Augmentation.

Not novel -- replicates what Henkel & Widmer 2021 already do to generalize
beyond synthetic MSMD audio (necessary before any real-audio evaluation
tier is meaningful).

CB_TA-Ext.md's IR_SOURCES (Aachen AIR, MIT McDermott Survey, OpenAIR) are
real academic impulse-response databases, but compute nodes on this cluster
have no internet access, and fetching exact download URLs for those
datasets isn't something to guess at (stale mirrors/redirects risk pulling
the wrong thing entirely). This generates SYNTHETIC room impulse responses
instead -- exponentially-decaying filtered noise, a standard substitute
when a real IR bank isn't available (used e.g. in several ASR reverberant
data-augmentation pipelines) -- clearly a stand-in for the real thing, not
a faithful reproduction of it. If real IR files become available later
(e.g. staged manually onto the cluster), swap load_ir_bank's synthetic
generator for a directory of real .wav files with the same call signature.
"""
from __future__ import annotations

import numpy as np


def synthesize_ir(sr: int, duration_sec: float, decay_tau_sec: float, seed: int | None = None) -> np.ndarray:
    """Exponentially-decaying filtered white noise, normalized to unit peak.
    Longer decay_tau_sec ~ a larger/more reverberant room."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration_sec)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n).astype(np.float32)
    envelope = np.exp(-t / decay_tau_sec).astype(np.float32)
    ir = noise * envelope
    # direct-path spike at t=0, louder than the diffuse tail
    ir[0] += 3.0
    ir = ir / (np.abs(ir).max() + 1e-8)
    return ir.astype(np.float32)


def build_synthetic_ir_bank(sr: int, n_irs: int = 16, tau_range=(0.1, 0.8), seed: int = 0):
    """A small bank of synthetic IRs spanning small-room to hall-like decay
    times, so augmented audio sees a variety of reverberant conditions
    rather than a single fixed one."""
    rng = np.random.default_rng(seed)
    taus = rng.uniform(tau_range[0], tau_range[1], size=n_irs)
    return [synthesize_ir(sr, duration_sec=min(2.0, 4 * tau), decay_tau_sec=tau, seed=int(seed_i))
            for seed_i, tau in enumerate(taus)]


def generate_pink_noise(n: int, seed: int | None = None) -> np.ndarray:
    """Simple pink-noise approximation via 1/f spectral shaping of white noise."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0   # avoid divide-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    return (pink / (np.abs(pink).max() + 1e-8)).astype(np.float32)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power = np.mean(signal ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    return signal + scaled_noise


def normalize_to_original_rms(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_rms = np.sqrt(np.mean(reference ** 2)) + 1e-12
    x_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    return x * (ref_rms / x_rms)


def apply_random_ir_augmentation(waveform: np.ndarray, ir_bank: list, sr: int, rng: np.random.Generator,
                                  p: float = 0.5, snr_range_db=(10, 30)) -> np.ndarray:
    """waveform: (num_samples,) synthetic MSMD audio, mono. Applies a random
    IR convolution + pink noise at a random SNR with probability p; returns
    the original waveform unchanged otherwise (so augmentation is stochastic
    per-sample, not applied to every training example)."""
    if rng.random() > p:
        return waveform
    from scipy.signal import fftconvolve
    ir = ir_bank[rng.integers(0, len(ir_bank))]
    convolved = fftconvolve(waveform, ir, mode='same').astype(np.float32)
    convolved = normalize_to_original_rms(convolved, waveform)
    noise = generate_pink_noise(len(waveform), seed=int(rng.integers(0, 2**31)))
    snr_db = rng.uniform(*snr_range_db)
    return mix_at_snr(convolved, noise, snr_db)
