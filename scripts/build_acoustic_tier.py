"""Build an ACOUSTIC-SHIFT test tier: the MSMD test pieces rendered with a
different piano, in a room, with recording noise -- then evaluated through
CPJKU's own `real_perf=True` audio path.

WHY THIS EXISTS. Every number in this project so far comes from MSMD audio
synthesised at test time with the SAME soundfont the models trained on
(`grand-piano-YDP-20160804.sf2`), through the same `midi_to_spec_otf` call.
Train and test therefore share an acoustic domain exactly, and no result so
far says anything about what happens when the audio is not that. Two things
depend on closing that gap:

  * B6 (`extensions/augmentation/impulse_response.py`) is an
    impulse-response augmentation built specifically to survive real
    recordings. CB_TA-Ext.md is explicit: "This ablation is judged on
    real-audio tiers only; synthetic MSMD performance is not the point."
    B6 has only ever been scored on synthetic MSMD (87.5%), so its entire
    purpose is currently untested.
  * B1a/MERT is pretrained on real music. Whether that pretraining also
    narrows the synthetic->real gap -- not just raises in-domain accuracy --
    is a much stronger claim than +4.7 points on synthetic data, and it is
    testable here.

WHY NOT ASAP (yet). ASAP is cloned at /scratch/pmohseni/datasets/asap-dataset
and gives real Disklavier performance MIDI + MusicXML + beat annotations. It
does NOT ship audio (that is MAESTRO, ~120GB separately) and, more
fundamentally, it has no rendered sheet images and no notehead PIXEL
coordinates. The score `.npz` this model consumes needs
`sheet (H,W) uint8`, `coords (N,3) [y,x,height]` and `coord2onset`, which
means rendering MusicXML and recovering per-notehead pixel positions (a
Verovio-SVG-style pipeline) before ASAP can be evaluated at all. That is a
separate multi-day build. This tier needs none of it: it reuses MSMD's
existing sheet/coords untouched and changes ONLY the audio, which is exactly
the variable under test.

WHAT IT PRODUCES (the layout `real_perf=True` expects, see
audio_conditioned_unet/utils.py:load_performance):
    <out>/score                -> symlink to msmd_test/score   (no copy: /project
                                  is at 96% of its inode quota)
    <out>/performance/{piece}.mid          timing source for onsets
    <out>/performance/{piece}_{tempo}.wav  the acoustically shifted audio

The MIDI is the tempo-`{tempo}` performance MIDI renamed, because real_perf
reads onsets from `{piece}.mid` while reading audio from
`{piece}_{tempo}.wav` -- the two must describe the same timeline or every
onset label is silently wrong.

DEGRADATIONS (all optional, each isolates a different domain shift):
  --soundfont   different piano => timbre shift
  --ir          convolutional reverb => room shift   (B6's own IR bank)
  --snr-db      pink noise        => recording shift (B6's own noise code)

Usage:
    python -m scripts.build_acoustic_tier --out /scratch/pmohseni/msmd_test_acoustic \
        --soundfont /path/to/other.sf2 --ir --snr-db 20
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

SRC_DEFAULT = os.path.join(_ROOT, 'third_party', 'cpjku_unet', 'data', 'msmd', 'msmd_test')
TRAIN_SF = os.path.join(_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet',
                        'sound_fonts', 'grand-piano-YDP-20160804.sf2')
SR = 22050


def synth(midi_path: str, wav_path: str, soundfont: str, sr: int = SR) -> None:
    """MIDI -> WAV via the fluidsynth CLI. `-ni` = no shell, no MIDI-in; `-F`
    renders to file rather than a device (there is no audio device on a
    compute node, hence also `-a file`)."""
    cmd = ['fluidsynth', '-ni', '-a', 'file', '-F', wav_path, '-r', str(sr),
           '-g', '0.8', soundfont, midi_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(wav_path):
        raise RuntimeError(f'fluidsynth failed for {midi_path}:\n{res.stderr[-800:]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC_DEFAULT, help='source msmd_test dir')
    ap.add_argument('--out', required=True, help='output tier dir (put on /scratch)')
    ap.add_argument('--tempo', type=int, default=1000)
    ap.add_argument('--soundfont', default=TRAIN_SF,
                    help='sf2 to render with; default is the TRAINING soundfont '
                         '(i.e. no timbre shift -- use a different one to shift)')
    ap.add_argument('--ir', action='store_true', help='apply convolutional reverb (B6 IR bank)')
    ap.add_argument('--ir-tau', type=float, default=0.45, help='reverb decay constant (s)')
    ap.add_argument('--snr-db', type=float, default=None, help='add pink noise at this SNR')
    ap.add_argument('--limit', type=int, default=None, help='only N pieces (smoke test)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    import soundfile as sf
    from scipy.signal import fftconvolve
    from extensions.augmentation.impulse_response import synthesize_ir, generate_pink_noise, \
        mix_at_snr, normalize_to_original_rms

    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, 'performance'), exist_ok=True)

    # score/ is symlinked, never copied: each .npz holds a full sheet image and
    # /project is at 96% of its 500k inode quota.
    score_link = os.path.join(a.out, 'score')
    if not os.path.exists(score_link):
        os.symlink(os.path.abspath(os.path.join(a.src, 'score')), score_link)

    pieces = sorted(os.path.basename(p)[:-4]
                    for p in os.listdir(os.path.join(a.src, 'score')) if p.endswith('.npz'))
    if a.limit:
        pieces = pieces[:a.limit]

    ir = synthesize_ir(SR, duration_sec=1.0, decay_tau_sec=a.ir_tau, seed=a.seed) if a.ir else None

    print(f'building acoustic tier -> {a.out}', flush=True)
    print(f'  pieces={len(pieces)} tempo={a.tempo} soundfont={os.path.basename(a.soundfont)} '
          f'ir={a.ir} snr_db={a.snr_db}', flush=True)

    ok = skipped = 0
    for i, piece in enumerate(pieces, 1):
        src_mid = os.path.join(a.src, 'performance', f'{piece}_tempo_{a.tempo}.mid')
        if not os.path.exists(src_mid):
            print(f'  [skip] no tempo-{a.tempo} MIDI for {piece}', flush=True)
            skipped += 1
            continue

        # real_perf reads onsets from '{piece}.mid' and audio from
        # '{piece}_{tempo}.wav'; both must describe the SAME timeline.
        dst_mid = os.path.join(a.out, 'performance', f'{piece}.mid')
        if not os.path.exists(dst_mid):
            shutil.copyfile(src_mid, dst_mid)

        dst_wav = os.path.join(a.out, 'performance', f'{piece}_{a.tempo}.wav')
        if os.path.exists(dst_wav):
            ok += 1
            continue

        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, 'raw.wav')
            synth(src_mid, raw, a.soundfont)
            x, sr = sf.read(raw, dtype='float32', always_2d=False)

        if x.ndim > 1:
            x = x.mean(axis=1)
        clean = x.copy()

        if ir is not None:
            x = fftconvolve(x, ir, mode='full')[:len(clean)]
            x = normalize_to_original_rms(x, clean)
        if a.snr_db is not None:
            noise = generate_pink_noise(len(x), seed=int(rng.integers(1 << 30)))
            x = mix_at_snr(x, noise, a.snr_db)
            x = normalize_to_original_rms(x, clean)

        peak = float(np.abs(x).max())
        if peak > 1.0:
            x = x / peak * 0.98
        sf.write(dst_wav, x.astype(np.float32), sr)
        ok += 1
        if i % 10 == 0 or i == len(pieces):
            print(f'  [{i}/{len(pieces)}] {piece}', flush=True)

    print(f'\ndone: {ok} pieces written, {skipped} skipped -> {a.out}', flush=True)
    print('evaluate with --test_dir {} and a config whose real_perf is True'.format(a.out), flush=True)


if __name__ == '__main__':
    main()
