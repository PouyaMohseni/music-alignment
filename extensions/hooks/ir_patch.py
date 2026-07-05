"""B6 -- monkey-patch midi_to_spec_otf to apply impulse-response augmentation
to the synthesized audio before computing the spectrogram.

Unlike B1a's load_piece patch, this doesn't need the piece/tempo identity
(augmentation is a fresh random decision each call, not a lookup keyed by
piece), so midi_to_spec_otf itself can be patched directly: it's called from
load_performance via a same-module reference (both defined in utils.py), so
patching audio_conditioned_unet.utils.midi_to_spec_otf is resolved fresh at
call time the same way load_piece's patch is -- and correctly reaches
spawned multiprocessing workers since it's an ordinary importable function,
not an in-memory stub.
"""
from __future__ import annotations
import os

import numpy as np

from extensions.augmentation.impulse_response import build_synthetic_ir_bank, apply_random_ir_augmentation

_IR_BANK = None            # built lazily -- fluidsynth's actual output sample
_IR_BANK_SR = None         # rate isn't 22050 (spec_params['sample_rate'], used
_N_IRS = 16                # only by the spectrogram resampler) -- it's whatever
_SEED = 0                  # fluidsynth's own default is (44100, confirmed by a
_RNG = None                # direct render check), and no -r flag overrides it.
_AUG_PROB = 0.5
_SNR_RANGE = (10.0, 30.0)


def _patched_midi_to_spec_otf(midi, spec_params, sound_font_path=None):
    import tempfile, time
    from audio_conditioned_unet.utils import spectrogram_processor, render_audio
    import scipy.io.wavfile as wavfile

    processor = spectrogram_processor(spec_params)
    mid_path = os.path.join(tempfile.gettempdir(), f'{os.getpid()}_{time.time()}.mid')
    with open(mid_path, 'wb') as f:
        midi.save(f)

    audio_path = render_audio(mid_path, sound_font=sound_font_path)

    sr, wav = wavfile.read(audio_path)
    wav_f = wav.astype(np.float32) / 32768.0
    if wav_f.ndim > 1:
        wav_f = wav_f.mean(axis=1)   # mono-mix if fluidsynth wrote stereo

    global _IR_BANK, _IR_BANK_SR
    if _IR_BANK is None or _IR_BANK_SR != sr:
        _IR_BANK = build_synthetic_ir_bank(sr=sr, n_irs=_N_IRS, seed=_SEED)
        _IR_BANK_SR = sr

    augmented = apply_random_ir_augmentation(wav_f, _IR_BANK, sr, _RNG, p=_AUG_PROB, snr_range_db=_SNR_RANGE)
    augmented = np.clip(augmented, -1.0, 1.0)
    wavfile.write(audio_path, sr, (augmented * 32767.0).astype(np.int16))

    spec = processor.process(audio_path).T

    os.remove(mid_path)
    os.remove(audio_path)
    return spec


def patch_ir_pipeline(p: float = 0.5, snr_range_db=(10.0, 30.0), n_irs: int = 16, seed: int = 0):
    global _RNG, _AUG_PROB, _SNR_RANGE, _N_IRS, _SEED, _IR_BANK, _IR_BANK_SR
    from audio_conditioned_unet import utils as cpjku_utils

    _RNG = np.random.default_rng(seed)
    _AUG_PROB = p
    _SNR_RANGE = snr_range_db
    _N_IRS = n_irs
    _SEED = seed
    _IR_BANK = None      # force rebuild at the actual fluidsynth output sample rate
    _IR_BANK_SR = None

    cpjku_utils.midi_to_spec_otf = _patched_midi_to_spec_otf
    print(f'[ir_patch] Patched midi_to_spec_otf with synthetic IR augmentation '
          f'(p={p}, snr_range={snr_range_db}, n_irs={n_irs})', flush=True)
