"""Build pitch-roll (88-key) supervision targets from MIDI ground truth.

Shared by v4 data loading and eval. These are TRAINING targets only — the
trained model takes (image, audio) at inference and never reads MIDI, so the
system stays end-to-end. We exploit the noiseless MIDI (audio is FluidSynth-
from-this-MIDI; score noteheads carry midi_pitch) purely to shape features.
"""
from __future__ import annotations
import numpy as np

MIDI_LOW = 21    # A0
N_PITCH = 88


def audio_pitchroll(onset_sec, offset_sec, pitch, T, eff_hz) -> np.ndarray:
    """(T, 88) float32 — pitch active from onset frame to offset frame."""
    pr = np.zeros((T, N_PITCH), dtype=np.float32)
    for on, off, p in zip(onset_sec, offset_sec, pitch):
        k = int(p) - MIDI_LOW
        if k < 0 or k >= N_PITCH:
            continue
        a = max(0, int(round(float(on) * eff_hz)))
        b = min(T, int(round(float(off) * eff_hz)) + 1)
        if b <= a:
            b = min(T, a + 1)
        pr[a:b, k] = 1.0
    return pr


def score_pitchroll(strip_x, pitch, tile_centers_px, tile_size) -> np.ndarray:
    """(N, 88) float32 — pitch active in every tile whose receptive field
    (width tile_size, centered at tile_centers_px[n]) contains the notehead."""
    N = len(tile_centers_px)
    pr = np.zeros((N, N_PITCH), dtype=np.float32)
    half = tile_size / 2.0
    centers = np.asarray(tile_centers_px, dtype=np.float64)
    for x, p in zip(strip_x, pitch):
        k = int(p) - MIDI_LOW
        if k < 0 or k >= N_PITCH:
            continue
        pr[np.abs(centers - float(x)) <= half, k] = 1.0
    return pr
