"""Stage 2 — synthesize audio.wav from score.midi.

Two backends are supported:

  - "fluidsynth" (preferred): shells out to the fluidsynth binary with a
    SoundFont. Produces realistic piano audio. Requires the fluidsynth CLI
    and an .sf2 file.

  - "pretty_midi" (fallback): pure-Python additive-sine synthesis via
    pretty_midi.synthesize(). Same timing as fluidsynth but the timbre is a
    sine bank; only useful for smoke-testing the pipeline. Does NOT require
    fluidsynth.

The output WAV is normalised so its peak amplitude sits at `peak_db` dBFS.

This module also re-runs the JSON sidecar update so audio.sha256, duration,
sample_rate_hz, soundfont, and peak_db are accurate post-synthesis.
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, tempfile, wave
import numpy as np

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_PEAK_DB     = -3.0


# ---------------------------------------------------------------- file I/O ---


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_wav_mono(path: str, samples: np.ndarray, sample_rate: int):
    """Write a mono float32 array in [-1, 1] to a 16-bit PCM WAV."""
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV (mono or stereo) into a float32 mono array."""
    with wave.open(path, "rb") as r:
        n_ch  = r.getnchannels()
        n_fr  = r.getnframes()
        sr    = r.getframerate()
        sw    = r.getsampwidth()
        raw   = r.readframes(n_fr)
    dtype = {1: "i1", 2: "<i2", 4: "<i4"}[sw]
    pcm   = np.frombuffer(raw, dtype=dtype).reshape(-1, n_ch).astype(np.float32)
    if sw == 2:
        pcm /= 32768.0
    elif sw == 4:
        pcm /= 2147483648.0
    mono = pcm.mean(axis=1)
    return mono, sr


def _normalize_peak(samples: np.ndarray, peak_db: float) -> np.ndarray:
    target = 10 ** (peak_db / 20.0)
    peak   = float(np.abs(samples).max())
    if peak < 1e-9:
        return samples
    return samples * (target / peak)


# ---------------------------------------------------------------- backends ---


def _synth_fluidsynth(midi_path: str, sf2_path: str, sample_rate: int) -> np.ndarray:
    """Call the fluidsynth CLI to render MIDI to a temp WAV, then load it."""
    fluid = shutil.which("fluidsynth")
    if fluid is None:
        raise RuntimeError("fluidsynth CLI not found on PATH")
    sf2_path = os.path.expanduser(sf2_path)   # expand ~ on all platforms
    midi_path = os.path.expanduser(midi_path)
    if not os.path.isfile(sf2_path):
        raise FileNotFoundError(f"soundfont not found: {sf2_path}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_tmp = tmp.name
    try:
        cmd = [
            fluid, "-ni",
            "-r", str(sample_rate),
            "-g", "1.0",                 # gain
            "-F", out_tmp,
            sf2_path, midi_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        samples, sr = _read_wav_mono(out_tmp)
        assert sr == sample_rate, f"fluidsynth wrote sr={sr}, expected {sample_rate}"
        return samples
    finally:
        if os.path.exists(out_tmp):
            os.remove(out_tmp)


def _synth_pretty_midi(midi_path: str, sample_rate: int) -> np.ndarray:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    return pm.synthesize(fs=sample_rate).astype(np.float32)


# ---------------------------------------------------------------- pipeline ---


def synthesize_audio(
    midi_path: str,
    out_wav: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sf2_path: str | None = None,
    peak_db: float = DEFAULT_PEAK_DB,
) -> dict:
    """Render midi_path → out_wav. Returns metadata dict for the JSON sidecar."""
    if sf2_path:
        samples  = _synth_fluidsynth(midi_path, sf2_path, sample_rate)
        soundfont = os.path.basename(sf2_path)
    else:
        samples  = _synth_pretty_midi(midi_path, sample_rate)
        soundfont = "pretty_midi_sine"

    samples = _normalize_peak(samples, peak_db)
    _write_wav_mono(out_wav, samples, sample_rate)

    return {
        "path":           os.path.basename(out_wav),
        "sha256":         _sha256(out_wav),
        "sample_rate_hz": sample_rate,
        "duration_sec":   len(samples) / sample_rate,
        "soundfont":      soundfont,
        "peak_db":        peak_db,
        "synthesized":    True,
    }


def update_annotations_audio(json_path: str, audio_block: dict) -> None:
    with open(json_path) as f:
        ann = json.load(f)
    ann["audio"] = audio_block
    with open(json_path, "w") as f:
        json.dump(ann, f, indent=2)


def synthesize_piece(
    piece_dir: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sf2_path: str | None = None,
    peak_db: float = DEFAULT_PEAK_DB,
) -> dict:
    """Run Stage 2 on one already-built piece directory.

    Reads score.midi, writes audio.wav, and updates annotations.json's audio block.
    """
    midi = os.path.join(piece_dir, "score.midi")
    wav  = os.path.join(piece_dir, "audio.wav")
    ann  = os.path.join(piece_dir, "annotations.json")
    block = synthesize_audio(midi, wav,
                             sample_rate=sample_rate, sf2_path=sf2_path, peak_db=peak_db)
    if os.path.exists(ann):
        update_annotations_audio(ann, block)
    return block
