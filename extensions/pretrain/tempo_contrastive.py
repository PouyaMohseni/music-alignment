"""C4 -- Tempo-invariant contrastive pretraining for CBEncoder.

MSMD renders every training piece at 7 tempo factors (500/750/950/1000/1050/
1250/1500 per configs/msmd_aug.yaml) -- the SAME note sequence, just played
back at a different speed. Verified directly on a real piece
(Anonymous__lesgraces__lesgraces_page_0): all 7 renders have identical note
COUNT and ORDER, with onset times scaled exactly by tempo_factor/1000
(e.g. tempo_500's last onset is exactly 0.5x tempo_1000's, tempo_1500's is
exactly 1.5x). This means the k-th onset event in any two tempo renders of
the same piece is the same musical moment -- no time-ratio arithmetic is
needed to find correspondences, just match onset events by sorted index.

That gives a free, dataset-native contrastive-pretraining signal: the
CBEncoder embedding of the audio window ending at onset k in tempo-A should
be close to the embedding of the audio window ending at onset k in tempo-B
(same musical content, different speed) and far from embeddings of other
onsets/pieces in the batch (different musical content). Standard InfoNCE.

Spectrogram computation reuses mymodel.cpjku_adapter.eval_official's
_wav_to_spec_logfilter (pure librosa reimplementation of madmom's exact
pipeline) rather than the real audio_conditioned_unet.utils spectrogram
path, so this runs in the main project venv (torch/librosa) without needing
real madmom or the venv_cpjku310 environment at all.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F

from mymodel.cpjku_adapter.eval_official import _wav_to_spec_logfilter

SPEC_PARAMS = {
    'sample_rate': 22050,
    'frame_size': 2048,
    'fps': 20,
    'pad': 40,
}
N_INPUT_FRAMES = 40   # CBEncoder.n_input_frames


def render_audio(midi_path: str, sound_font: str, fluidsynth_bin: str) -> str:
    out_wav = os.path.join(tempfile.gettempdir(), f'{os.getpid()}_{time.time()}.wav')
    cmd = [fluidsynth_bin, '-R', '0', '-C', '0', '-F', out_wav, '-O', 's16', '-T', 'wav',
           sound_font, midi_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_wav


def render_and_spec(midi_path: str, sound_font: str, fluidsynth_bin: str) -> np.ndarray:
    """-> (78, T+pad) float32, same convention CBEncoder's training spec uses."""
    wav_path = render_audio(midi_path, sound_font, fluidsynth_bin)
    try:
        return _wav_to_spec_logfilter(wav_path, SPEC_PARAMS)
    finally:
        os.remove(wav_path)


def get_onset_frame_indices(midi_path: str, fps: int = SPEC_PARAMS['fps'],
                             pad: int = SPEC_PARAMS['pad']) -> np.ndarray:
    """Sorted onset frame indices (into the PADDED spec) for every note in the
    MIDI file, across all instruments. Ties (chord notes at the same onset)
    keep a stable sort so index-based correspondence across two tempo
    renders of the same piece holds (verified: note count/order match
    exactly across tempo factors)."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    onset_secs = sorted(n.start for inst in midi.instruments for n in inst.notes)
    onset_frames = pad + np.round(np.array(onset_secs) * fps).astype(int)
    return onset_frames


def extract_window(spec: np.ndarray, frame_idx: int, n_input_frames: int = N_INPUT_FRAMES) -> np.ndarray:
    """spec[:, frame_idx-n_input_frames+1 : frame_idx+1], the exact windowing
    convention audio_conditioned_unet/dataset.py's __getitem__ uses
    (spec[:, i-self.n_frames+1:i+1])."""
    start = frame_idx - n_input_frames + 1
    end = frame_idx + 1
    window = spec[:, max(start, 0):end]
    if window.shape[1] < n_input_frames:
        # Only possible if frame_idx < n_input_frames-1, which pad=40 ==
        # n_input_frames=40 should already prevent for onset 0 (frame_idx =
        # pad+0 = 40, start = 40-39 = 1 >= 0) -- left-pad defensively anyway.
        window = np.pad(window, ((0, 0), (n_input_frames - window.shape[1], 0)), mode='edge')
    return window


def sample_tempo_pair_batch(piece_stem: str, perf_dir: str, tempo_a: int, tempo_b: int,
                             sound_font: str, fluidsynth_bin: str, batch_size: int,
                             rng: np.random.Generator):
    """Returns (batch_a, batch_b), each a (1, B, 1, 78, 40) float32 array
    ready for CBEncoder's (seq_len, bs, c, h, w) input convention, plus the
    number of onsets actually available (for logging/skip decisions)."""
    midi_a = Path(perf_dir) / f'{piece_stem}_tempo_{tempo_a}.mid'
    midi_b = Path(perf_dir) / f'{piece_stem}_tempo_{tempo_b}.mid'

    spec_a = render_and_spec(str(midi_a), sound_font, fluidsynth_bin)
    spec_b = render_and_spec(str(midi_b), sound_font, fluidsynth_bin)

    onsets_a = get_onset_frame_indices(midi_a)
    onsets_b = get_onset_frame_indices(midi_b)
    n_common = min(len(onsets_a), len(onsets_b))
    if n_common == 0:
        return None, None, 0

    k = min(batch_size, n_common)
    idx = rng.choice(n_common, size=k, replace=False)

    windows_a = np.stack([extract_window(spec_a, onsets_a[i]) for i in idx])   # (B,78,40)
    windows_b = np.stack([extract_window(spec_b, onsets_b[i]) for i in idx])

    batch_a = windows_a[np.newaxis, :, np.newaxis].astype(np.float32)   # (1,B,1,78,40)
    batch_b = windows_b[np.newaxis, :, np.newaxis].astype(np.float32)
    return batch_a, batch_b, k


def info_nce_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE: emb_a[i] should match emb_b[i] (same onset, two
    tempos) against all other emb_b[j] (and vice versa) in the batch."""
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)
    logits = emb_a @ emb_b.t() / temperature   # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) / 2
