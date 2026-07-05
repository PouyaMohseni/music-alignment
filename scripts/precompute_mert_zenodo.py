"""Precompute MERT-v1-95M embeddings for the Zenodo CB_TA dataset (A0's own
train/val/test data), one embedding sequence per (piece, tempo_factor) MIDI.

This lets B1 (frozen MERT audio-encoder swap) reuse CPJKU's unmodified
train_model.py / ConditionalUNet / dataset.py: we monkey-patch
midi_to_spec_otf/wav_to_spec_otf (see extensions/hooks/mert_patch.py) to load
these cached (768, T_20fps) arrays instead of computing a live mel-spectrogram
-- everything downstream (padding, windowing, onset alignment) is unchanged
since it only depends on array shape, not content.

Runs in the MAIN project venv (has torch/transformers/librosa); the CPJKU
venv (venv_cpjku310) has neither, so live MERT can't run there. FluidSynth
itself is invoked directly via subprocess (same command CPJKU's own
render_audio uses) rather than importing their utils, since that requires
real madmom which isn't installed in this venv either.

    python scripts/precompute_mert_zenodo.py \
        --midi_dir  /scratch/pmohseni/msmd_train_full/performance \
        --out_dir   /scratch/pmohseni/mert_emb_zenodo/train_full \
        --sound_font third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2 \
        --fluidsynth /scratch/pmohseni/micromamba/envs/fluidsynth/bin/fluidsynth
"""
from __future__ import annotations
import argparse, os, subprocess, tempfile, time
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.interpolate import interp1d
from transformers import AutoModel

MERT_SR  = 24000
MERT_FPS = 75    # native output frame rate of MERT-v1-95M


def _load_model(model_id, device):
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def render_audio(midi_path: str, sound_font: str, fluidsynth_bin: str, out_wav: str):
    cmd = [fluidsynth_bin, "-R", "0", "-C", "0", "-F", out_wav, "-O", "s16", "-T", "wav",
           sound_font, midi_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


@torch.no_grad()
def encode_wav(model, wav_path, chunk_sec=10.0, device='cuda'):
    y, _ = librosa.load(str(wav_path), sr=MERT_SR, mono=True)
    chunk = int(chunk_sec * MERT_SR)
    outs = []
    for s in range(0, len(y), chunk):
        seg = y[s:s + chunk]
        if len(seg) < MERT_SR // 4:
            break
        t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(device)
        out = model(input_values=t).last_hidden_state[0]
        outs.append(out.cpu().float().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 768), dtype=np.float32)


def resample_emb(emb, src_fps, dst_fps):
    T = emb.shape[0]
    if T == 0:
        return emb
    times_src = np.arange(T) / src_fps
    T_dst = max(1, int(round(T * dst_fps / src_fps)))
    times_dst = np.clip(np.arange(T_dst) / dst_fps, 0, times_src[-1])
    f = interp1d(times_src, emb, axis=0, bounds_error=False, fill_value='extrapolate')
    return f(times_dst).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--midi_dir', required=True, help='dir of .mid files to encode')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--sound_font', required=True)
    p.add_argument('--fluidsynth', required=True)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--mert_id', default='m-a-p/MERT-v1-95M')
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading MERT ({a.mert_id}) on {device}...', flush=True)
    model = _load_model(a.mert_id, device)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(Path(a.midi_dir).glob('*.mid'))
    print(f'{len(midi_files)} MIDI files to encode', flush=True)

    done = skip = fail = 0
    for i, midi_path in enumerate(midi_files):
        key = midi_path.stem   # e.g. Anonymous__..._page_0_tempo_1000
        out_path = out_dir / f'{key}.npy'
        if out_path.exists():
            skip += 1
            continue
        wav_path = os.path.join(tempfile.gettempdir(), f'{os.getpid()}_{time.time()}.wav')
        try:
            render_audio(str(midi_path), a.sound_font, a.fluidsynth, wav_path)
            emb = encode_wav(model, wav_path, device=device)
            if emb.shape[0] == 0:
                print(f'  SKIP {key}: empty audio', flush=True)
                fail += 1
                continue
            emb20 = resample_emb(emb, MERT_FPS, a.fps)
            np.save(out_path, emb20.astype(np.float16))
            done += 1
        except Exception as e:
            print(f'  FAIL {key}: {e}', flush=True)
            fail += 1
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(midi_files)}] done={done} skip={skip} fail={fail}', flush=True)

    print(f'Done. done={done} skip={skip} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
