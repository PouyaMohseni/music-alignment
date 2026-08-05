"""H1 -- MERT embeddings for CYOLO's OWN audio, on CYOLO's OWN frame grid.

WHY A THIRD BANK. The existing banks (mert_emb_zenodo, mert_emb_aug) are keyed
by MSMD MIDI stems, `{piece}_page_{n}_tempo_{tf}`, and were rendered by us
through fluidsynth. CYOLO does not use those: dataset.py reads ONE
`{piece_name}.wav` per piece from its own data directory
(utils/data_utils.py:34-35), where `piece_name = basename(score_path)[:-4]`.
Different audio, different keys, so nothing can be reused.

WHY THE FRAME GRID IS THE WHOLE PROBLEM. CYOLO indexes performances by FRAME:

    start_t = int(start_frame * hop_length)
    t       = FRAME_SIZE + int(frame * hop_length)
    truncated_signal = signal[start_t:t]                 (dataset.py:70-76)

and its FPS is **SAMPLE_RATE / HOP_SIZE = 22050 / 1102 = 20.0091**, NOT 20.
Our other banks are resampled to exactly 20.0. Substituting them would drift by
~1.6 frames over a 3-minute piece -- under the 10-frame (0.5 s) tolerance, so
it would not obviously break, it would just quietly cost accuracy on the
longest pieces and be nearly impossible to attribute later. So this script
resamples to CYOLO's exact rate.

Frame count is matched to what LogSpectrogram would have produced for the same
signal, `1 + (len(y) - FRAME_SIZE) // HOP_SIZE`, so MERT frame f stands in for
exactly the window CYOLO's frame f covered and every existing sequence index
stays valid without touching the sequence metadata.

Output: {out_dir}/{piece_name}.npy, float16 (T, 768).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from scripts.precompute_mert_zenodo import MERT_FPS, MERT_SR, _load_model, encode_wav

# CYOLO's constants (cyolo_score_following/utils/data_utils.py:11-14). Hardcoded
# rather than imported so this script does not need CYOLO's package (and its
# madmom/torch pins) importable in the MERT environment.
CY_SAMPLE_RATE = 22050
CY_FRAME_SIZE = 2048
CY_HOP_SIZE = 1102
CY_FPS = CY_SAMPLE_RATE / CY_HOP_SIZE          # 20.0091, deliberately not 20


def resample_to_n(emb: np.ndarray, src_fps: float, n_dst: int) -> np.ndarray:
    """Resample (T_src, D) to exactly n_dst frames on the destination grid.

    Resampling to a COUNT rather than to an fps: the count is what the sequence
    indices were built against, and letting it fall out of a float fps
    conversion is how off-by-one drift gets in.
    """
    from scipy.interpolate import interp1d
    t_src = emb.shape[0]
    if t_src == 0:
        return np.zeros((n_dst, emb.shape[1]), dtype=np.float32)
    if t_src == 1:
        return np.repeat(emb, n_dst, axis=0).astype(np.float32)
    times_src = np.arange(t_src) / src_fps
    times_dst = np.clip(np.arange(n_dst) / CY_FPS, 0.0, times_src[-1])
    f = interp1d(times_src, emb, axis=0, bounds_error=False, fill_value='extrapolate')
    return f(times_dst).astype(np.float32)


def n_frames_for(n_samples: int) -> int:
    """Frames LogSpectrogram yields for a signal of this length, so MERT frame f
    covers the window CYOLO frame f covered."""
    return max(1, 1 + (n_samples - CY_FRAME_SIZE) // CY_HOP_SIZE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wav_dir', required=True, help="CYOLO split dir holding {piece}.wav")
    p.add_argument('--out_dir', required=True)
    p.add_argument('--mert_id', default='m-a-p/MERT-v1-95M')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    a = p.parse_args()

    import soundfile as sf

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading MERT ({a.mert_id}) on {device}...', flush=True)
    model = _load_model(a.mert_id, device)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(Path(a.wav_dir).glob('*.wav'))
    mine = wavs[a.shard::a.num_shards]
    print(f'shard {a.shard}/{a.num_shards}: {len(mine)} of {len(wavs)} wavs from {a.wav_dir}',
          flush=True)
    print(f'  target grid: fps={CY_FPS:.6f} (={CY_SAMPLE_RATE}/{CY_HOP_SIZE}), '
          f'frame_size={CY_FRAME_SIZE}', flush=True)

    done = skip = fail = 0
    for i, wav_path in enumerate(mine):
        key = wav_path.stem
        out_path = out_dir / f'{key}.npy'
        if out_path.exists():
            skip += 1
            continue
        try:
            info = sf.info(str(wav_path))
            # length in CYOLO's own sample rate, which is what its frame indices assume
            n_samples = int(round(info.frames * CY_SAMPLE_RATE / info.samplerate))
            n_dst = n_frames_for(n_samples)

            emb = encode_wav(model, str(wav_path), device=device)     # (T, 768) @ MERT_FPS
            if emb.shape[0] == 0:
                print(f'  SKIP {key}: empty audio', flush=True)
                fail += 1
                continue
            # Durations must agree: MERT saw the whole file, and n_dst was derived
            # from that same file's length. If they disagree the wav header lied
            # about its rate or the file is truncated -- in which case
            # resample_to_n's clipping would silently flat-line the tail into a
            # repeated last frame instead of failing, and training would consume
            # a constant embedding for the end of the piece.
            dur_src = emb.shape[0] / MERT_FPS
            dur_dst = n_dst / CY_FPS
            if abs(dur_src - dur_dst) > max(0.5, 0.02 * dur_dst):
                raise ValueError(f'duration mismatch: MERT {dur_src:.2f}s vs '
                                 f'CYOLO grid {dur_dst:.2f}s ({n_dst} frames)')

            out = resample_to_n(emb, MERT_FPS, n_dst)
            assert out.shape == (n_dst, 768), f'{key}: {out.shape} != {(n_dst, 768)}'
            np.save(out_path, out.astype(np.float16))
            done += 1
        except Exception as e:
            print(f'  FAIL {key}: {type(e).__name__}: {e}', flush=True)
            fail += 1
        if (i + 1) % 25 == 0 or i + 1 == len(mine):
            print(f'  [{i+1}/{len(mine)}] done={done} skip={skip} fail={fail}', flush=True)

    print(f'Done shard {a.shard}. done={done} skip={skip} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
