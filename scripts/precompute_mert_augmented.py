"""R2 -- re-encode the MSMD training set with MERT after ACOUSTIC degradation,
producing a second embedding bank that training can mix in.

WHY THIS AND NOT B6. B6 (extensions/augmentation/impulse_response.py) already
tried impulse-response augmentation and came LAST on the tier it was designed
for (15.6% pct@0.5s on `room`). Two things were wrong with it, and both are
fixed here:

  1. WRONG BRANCH. B6 augmented the CBEncoder path. The 2026-08-03 sweep shows
     essentially all of the real-audio headroom is on the MERT path: every
     MERT model scores 37-44 on `room` while every CBEncoder model scores
     15-28. Augmenting the branch that was already losing by 20 points could
     not have produced a competitive real-audio model.
  2. WRONG NUISANCE. B6 applied reverb (and optionally pink noise) only.
     Reverb is a *temporal* smearing. The dominant difference between
     fluidsynth output and a room mic is a *static per-band gain* -- the
     combined mic response, room transfer function and distance -- which
     reverb alone does not reproduce. `--tilt` adds exactly that, drawn fresh
     per rendering, and it is the degradation the R1 analysis says matters.

WHY IT NEEDS A SEPARATE EMBEDDING BANK AT ALL. MERT is frozen and consumed
from precomputed .npy (extensions/hooks/mert_patch.py never encodes audio), so
waveform augmentation is invisible unless the augmented audio is pushed
through MERT ahead of time. Hence 6615 renders rather than an on-the-fly
transform.

NO TEST LEAKAGE. Every degradation parameter is drawn from a fixed prior with
a seed derived from the piece key. Nothing here is fitted to, or measured
from, MSMD-Rec.

Key names match the clean bank exactly ({piece}_tempo_{tf}.npy) so a training
hook can swap one root for the other per sample without any renaming.

Sharded for a SLURM array:
    python -m scripts.precompute_mert_augmented --midi_dir ... --out_dir ... \
        --sound_font ... --fluidsynth ... --shard $SLURM_ARRAY_TASK_ID --num_shards 40
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from scripts.precompute_mert_zenodo import (MERT_FPS, MERT_SR, _load_model,
                                            encode_wav, render_audio, resample_emb)


def _seed_for(key: str, salt: int) -> int:
    """Deterministic per-piece seed, so a re-run of a failed shard reproduces
    the same degradation instead of silently creating a second condition."""
    h = hashlib.sha1(f'{key}:{salt}'.encode()).hexdigest()
    return int(h[:8], 16)


def random_tilt(y: np.ndarray, sr: int, rng: np.random.Generator,
                max_db: float = 12.0, n_modes: int = 4) -> np.ndarray:
    """Apply a smooth random per-frequency gain -- a stand-in for the combined
    mic + room + distance transfer function.

    The curve is a sum of a few low-order cosines in log-frequency, so it is
    smooth (a real channel does not have per-bin structure) and can reach
    +/- max_db. Applied by STFT magnitude multiplication with the original
    phase kept, which is a zero-phase filter -- adequate here because the
    downstream feature is a magnitude representation.
    """
    n_fft, hop = 2048, 512
    stft = np.fft.rfft(_frame(y, n_fft, hop) * np.hanning(n_fft)[None, :], axis=1)
    n_bins = stft.shape[1]

    log_f = np.linspace(0.0, 1.0, n_bins)
    curve_db = np.zeros(n_bins)
    for k in range(1, n_modes + 1):
        curve_db += rng.normal() * np.cos(np.pi * k * log_f) / k
    # normalise then scale, so max_db means what it says regardless of n_modes
    peak = np.abs(curve_db).max()
    if peak > 1e-8:
        curve_db = curve_db / peak * rng.uniform(0.3, 1.0) * max_db

    gain = 10.0 ** (curve_db / 20.0)
    out = _overlap_add(np.fft.irfft(stft * gain[None, :], n=n_fft, axis=1)
                       * np.hanning(n_fft)[None, :], hop, len(y))
    return out.astype(np.float32)


def _frame(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    # CEIL, not floor: flooring drops up to hop-1 trailing samples, so the
    # reconstruction comes back shorter than the input and the tail of every
    # performance is silently truncated.
    n_frames = 1 + int(np.ceil(max(0, len(y) - n_fft) / hop))
    padded = np.pad(y, (0, max(0, n_fft + (n_frames - 1) * hop - len(y))))
    return np.stack([padded[i * hop:i * hop + n_fft] for i in range(n_frames)])


def _overlap_add(frames: np.ndarray, hop: int, out_len: int) -> np.ndarray:
    n_frames, n_fft = frames.shape
    out = np.zeros(n_fft + (n_frames - 1) * hop)
    wsum = np.zeros_like(out)
    w = np.hanning(n_fft) ** 2
    for i in range(n_frames):
        out[i * hop:i * hop + n_fft] += frames[i]
        wsum[i * hop:i * hop + n_fft] += w
    # Floor relative to the steady-state window sum: the first and last half
    # window are covered by only one frame, and dividing those by a near-zero
    # wsum would amplify the fade-in/out into a click.
    out = out / np.maximum(wsum, 1e-3 * wsum.max())
    if len(out) < out_len:
        out = np.pad(out, (0, out_len - len(out)))
    return out[:out_len]


def degrade(y: np.ndarray, sr: int, rng: np.random.Generator, args) -> np.ndarray:
    from scipy.signal import fftconvolve
    from extensions.augmentation.impulse_response import (generate_pink_noise, mix_at_snr,
                                                          normalize_to_original_rms,
                                                          synthesize_ir)
    clean = y.copy()

    if args.tilt and rng.random() < args.p_tilt:
        y = random_tilt(y, sr, rng, max_db=args.tilt_db)

    if args.ir and rng.random() < args.p_ir:
        # tau spread over a wide range of room sizes rather than B6's single
        # 0.45 s, and mixed dry/wet so "close mic" is inside the prior too.
        tau = float(rng.uniform(0.15, 0.9))
        ir = synthesize_ir(sr, duration_sec=min(1.5, 3 * tau), decay_tau_sec=tau,
                           seed=int(rng.integers(1 << 30)))
        wet = fftconvolve(y, ir, mode='full')[:len(y)]
        mix = float(rng.uniform(0.2, 0.9))
        y = (1.0 - mix) * y + mix * wet
        y = normalize_to_original_rms(y, clean)

    if args.noise and rng.random() < args.p_noise:
        snr = float(rng.uniform(args.snr_lo, args.snr_hi))
        y = mix_at_snr(y, generate_pink_noise(len(y), seed=int(rng.integers(1 << 30))), snr)
        y = normalize_to_original_rms(y, clean)

    y = y * float(rng.uniform(0.5, 1.4))          # recording level
    peak = float(np.abs(y).max())
    if peak > 1.0:
        y = y / peak * 0.98
    return y.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--midi_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--sound_font', required=True)
    p.add_argument('--fluidsynth', required=True)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--mert_id', default='m-a-p/MERT-v1-95M')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--salt', type=int, default=0, help='bump to build a 2nd independent condition')
    p.add_argument('--tilt', action='store_true'); p.add_argument('--p_tilt', type=float, default=0.9)
    p.add_argument('--tilt_db', type=float, default=12.0)
    p.add_argument('--ir', action='store_true'); p.add_argument('--p_ir', type=float, default=0.8)
    p.add_argument('--noise', action='store_true'); p.add_argument('--p_noise', type=float, default=0.6)
    p.add_argument('--snr_lo', type=float, default=12.0)
    p.add_argument('--snr_hi', type=float, default=35.0)
    a = p.parse_args()

    import librosa, soundfile as sf  # noqa: F401  (librosa needed by encode_wav)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading MERT ({a.mert_id}) on {device}...', flush=True)
    model = _load_model(a.mert_id, device)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(Path(a.midi_dir).glob('*.mid'))
    mine = midi_files[a.shard::a.num_shards]
    print(f'shard {a.shard}/{a.num_shards}: {len(mine)} of {len(midi_files)} MIDI files',
          flush=True)
    print(f'  tilt={a.tilt}({a.tilt_db}dB) ir={a.ir} noise={a.noise} '
          f'snr={a.snr_lo}-{a.snr_hi} salt={a.salt}', flush=True)

    done = skip = fail = 0
    for i, midi_path in enumerate(mine):
        key = midi_path.stem
        out_path = out_dir / f'{key}.npy'
        if out_path.exists():
            skip += 1
            continue
        rng = np.random.default_rng(_seed_for(key, a.salt))
        wav_path = os.path.join(tempfile.gettempdir(), f'{os.getpid()}_{time.time()}.wav')
        try:
            render_audio(str(midi_path), a.sound_font, a.fluidsynth, wav_path)
            y, sr = librosa.load(wav_path, sr=MERT_SR, mono=True)
            y = degrade(y, sr, rng, a)
            sf.write(wav_path, y, sr)

            emb = encode_wav(model, wav_path, device=device)
            if emb.shape[0] == 0:
                print(f'  SKIP {key}: empty audio', flush=True)
                fail += 1
                continue
            np.save(out_path, resample_emb(emb, MERT_FPS, a.fps).astype(np.float16))
            done += 1
        except Exception as e:
            print(f'  FAIL {key}: {e}', flush=True)
            fail += 1
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(mine)}] done={done} skip={skip} fail={fail}', flush=True)

    print(f'Done shard {a.shard}. done={done} skip={skip} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
