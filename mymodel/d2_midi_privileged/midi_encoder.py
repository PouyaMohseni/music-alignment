"""D2 component 2 -- MIDI-privileged cross-modal distillation.

MidiEncoder projects a per-frame active-pitch multi-hot vector (built from
whole-piece MIDI -- exact, zero-noise description of the sounding notes at
each instant) into the SAME embedding space as D1's audio tower. A symmetric
InfoNCE loss (mymodel/d2_midi_privileged/losses.py) pulls the audio tower
toward this privileged target during training. MidiEncoder is a pure training
scaffold: eval.py never imports it, so no MIDI ever touches inference.

Pitch-roll construction mirrors extensions/hooks/pitch_patch.py's
_compute_pitch_roll (same 88-key piano range, same onset/duration framing),
reimplemented against pretty_midi (already used by C4's tempo-contrastive
pretraining) instead of madmom's MIDI reader, so this runs in the main .venv.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PITCHES = 88
MIDI_PITCH_OFFSET = 21   # standard piano range 21-108 -> index 0-87


def compute_pitch_roll(midi_path: str, T_frames: int, fps: float) -> np.ndarray:
    """Returns (T_frames, 88) float32 multi-hot active-pitch roll from a
    whole-piece MIDI file, at the given frame rate."""
    import pretty_midi
    m = pretty_midi.PrettyMIDI(str(midi_path))
    roll = np.zeros((T_frames, NUM_PITCHES), dtype=np.float32)
    for inst in m.instruments:
        for n in inst.notes:
            onset_f = int(np.clip(round(n.start * fps), 0, T_frames - 1))
            end_f = int(np.clip(round(n.end * fps), 0, T_frames))
            p = int(np.clip(n.pitch - MIDI_PITCH_OFFSET, 0, NUM_PITCHES - 1))
            roll[onset_f:max(end_f, onset_f + 1), p] = 1.0
    return roll


class MidiEncoder(nn.Module):
    def __init__(self, d_model: int = 128, num_pitches: int = NUM_PITCHES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_pitches, 64), nn.GELU(),
            nn.Linear(64, d_model),
        )

    def forward(self, pitch_roll: torch.Tensor) -> torch.Tensor:
        """pitch_roll: (T, 88) -> (T, d_model) L2-normalized."""
        return F.normalize(self.net(pitch_roll), dim=-1)
