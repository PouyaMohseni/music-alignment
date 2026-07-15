"""Precompute REAL madmom spectrograms for cpjku_fmt performances.

Root cause this fixes: eval_official.py (run under the main .venv, which
cannot have real madmom on Python>=3.11) approximates madmom's spectrogram
with a hand-rolled librosa filterbank. Training (venv_cpjku310, real madmom)
does NOT use this approximation. That train/eval domain mismatch was proven
decisive by running the frozen, unmodified official CB_TA checkpoint through
both harnesses: ~85% via their own eval_model.py (real madmom, venv_cpjku310)
vs 15.1% via eval_official.py (librosa approximation, main .venv), on the
same MSMD test pieces.

This script must run under venv_cpjku310 (real madmom, Python 3.10). It
computes the exact same spectrogram training uses (wav_to_spec_otf, already
padded to match load_performance's convention) and caches it to
<cpjku_data>/spec_madmom/<piece_id>.npy. eval_official.py then loads this
cache instead of the librosa approximation, closing the gap for the frozen
official checkpoint AND for our own from-scratch cpjku_adapter experiments
(b1a-b6, c2, cpjku_aug), whose eval scripts hit the same approximation.

    module load gcc opencv
    source /scratch/pmohseni/venv_cpjku310/bin/activate
    python -m mymodel.cpjku_adapter.precompute_madmom_specs \
        --cpjku_root third_party/cpjku_unet --cpjku_data data/MSMD/cpjku_fmt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpjku_root', default='third_party/cpjku_unet')
    p.add_argument('--cpjku_data', default='data/MSMD/cpjku_fmt')
    p.add_argument('--sample_rate', type=int, default=22050)
    p.add_argument('--frame_size',  type=int, default=2048)
    p.add_argument('--fps',         type=int, default=20)
    p.add_argument('--pad',         type=int, default=40)
    a = p.parse_args()

    cpjku_root = Path(a.cpjku_root).resolve()
    if str(cpjku_root) not in sys.path:
        sys.path.insert(0, str(cpjku_root))

    import madmom  # noqa: F401 -- fail loudly here if not real madmom
    from audio_conditioned_unet.utils import wav_to_spec_otf
    import numpy as np

    spec_params = {
        'sample_rate': a.sample_rate,
        'frame_size':  a.frame_size,
        'fps':         a.fps,
        'pad':         a.pad,
    }

    cpjku_data = Path(a.cpjku_data)
    perf_dir = cpjku_data / 'performance'
    out_dir  = cpjku_data / 'spec_madmom'
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(perf_dir.glob('*.wav'))
    # skip the *_1000.wav duplicates (symlinked to the same audio.wav) -- only
    # need one spec per piece, keyed by the plain <piece_id>.wav name.
    wavs = [w for w in wavs if not w.stem.endswith('_1000')]

    print(f'Found {len(wavs)} performances to process', flush=True)
    done = skip = fail = 0
    for i, wav_path in enumerate(wavs):
        pid = wav_path.stem
        out_path = out_dir / f'{pid}.npy'
        if out_path.exists():
            skip += 1
            continue
        try:
            spec = wav_to_spec_otf(str(wav_path), spec_params)
            spec = np.pad(spec, ((0, 0), (spec_params['pad'], 0)), mode='constant')
            np.save(out_path, spec.astype(np.float32))
            done += 1
        except Exception as e:
            print(f'  FAIL {pid}: {e}', flush=True)
            fail += 1
        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(wavs)}] done={done} skip={skip} fail={fail}', flush=True)

    print(f'Done. done={done} skip={skip} fail={fail} total={len(wavs)}', flush=True)


if __name__ == '__main__':
    main()
