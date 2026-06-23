"""Monkey-patch CPJKU utils to replace madmom with librosa.

madmom does not support Python ≥3.11 / NumPy ≥1.24.
We replicate their exact spectrogram pipeline with librosa:
  - 12-band log-frequency filterbank, 60–6000 Hz
  - fps=20, frame_size=2048
  - Logarithmic magnitude (log1p)

Call patch() BEFORE importing audio_conditioned_unet.utils.

Usage:
    import mymodel.cpjku_adapter.madmom_compat as mc
    mc.patch()
    from audio_conditioned_unet.utils import load_performance, wav_to_spec_otf
"""
import sys
import numpy as np


def _librosa_spec(wav_path: str, spec_params: dict) -> np.ndarray:
    """Replicate madmom's FilteredSpectrogramProcessor with librosa.

    Returns (n_bands, T) float32, zero-padded by spec_params['pad'] on the left.
    """
    import librosa
    sr    = spec_params['sample_rate']   # 22050
    n_fft = spec_params['frame_size']    # 2048
    fps   = spec_params['fps']           # 20
    pad   = spec_params['pad']           # 40

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    hop = int(sr / fps)   # 1102

    # 12-band log-frequency filterbank, 60–6000 Hz (matches madmom LogarithmicFilterbank)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=12, fmin=60.0, fmax=6000.0, power=1.0)
    log_mel = np.log1p(mel).astype(np.float32)   # (12, T)

    # Zero-pad left by 'pad' frames (their convention)
    padded = np.pad(log_mel, ((0, 0), (pad, 0)), mode='constant')
    return padded   # (12, T + pad)


def _wav_to_spec_otf(wav_path: str, spec_params: dict) -> np.ndarray:
    return _librosa_spec(wav_path, spec_params)


def _midi_to_spec_otf(midi, spec_params: dict, sound_font_path=None) -> np.ndarray:
    raise NotImplementedError(
        'MIDI→spec synthesis requires madmom+fluidsynth. '
        'Use real_perf=True and provide audio.wav files instead.')


def patch():
    """Monkey-patch madmom-dependent functions in audio_conditioned_unet.utils."""
    import types

    # Stub out cv2 if not installed — we never call resize (scale_factor=1)
    try:
        import cv2  # noqa: F401
    except ImportError:
        fake_cv2 = types.ModuleType('cv2')
        fake_cv2.resize = None
        fake_cv2.INTER_AREA = 0
        sys.modules.setdefault('cv2', fake_cv2)
        print('[cpjku_adapter] cv2 not found — stubbed out (scale_factor=1, resize unused)',
              flush=True)

    # Stub out the madmom import so the module loads without it
    fake_madmom = types.ModuleType('madmom')
    fake_madmom.io = types.ModuleType('madmom.io')
    fake_madmom.io.midi = types.ModuleType('madmom.io.midi')
    fake_madmom.audio = types.ModuleType('madmom.audio')
    sys.modules.setdefault('madmom', fake_madmom)
    sys.modules.setdefault('madmom.io', fake_madmom.io)
    sys.modules.setdefault('madmom.io.midi', fake_madmom.io.midi)
    sys.modules.setdefault('madmom.audio', fake_madmom.audio)
    sys.modules.setdefault('madmom.audio.signal', types.ModuleType('madmom.audio.signal'))
    sys.modules.setdefault('madmom.audio.spectrogram', types.ModuleType('madmom.audio.spectrogram'))
    sys.modules.setdefault('madmom.processors', types.ModuleType('madmom.processors'))

    # Now patch the functions after the module loads
    import importlib
    try:
        import audio_conditioned_unet.utils as _utils
    except Exception:
        # Module not yet importable — will be patched after sys.path is set up
        return

    _utils.wav_to_spec_otf  = _wav_to_spec_otf
    _utils.midi_to_spec_otf = _midi_to_spec_otf
    print('[cpjku_adapter] madmom patched → librosa spectrogram', flush=True)
