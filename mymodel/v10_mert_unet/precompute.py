"""Precompute MERT-v1-95M embeddings at fps=20 for all MSMD pieces.

Saves per-piece float16 arrays:
    <out_dir>/<piece_id>.npy   shape (T_20hz, 768)

Runtime: ~30-60 min on A100 for all 454 MSMD pieces (train+val+test).

    python -m mymodel.v10_mert_unet.precompute \
        --processed data/MSMD/processed \
        --out       data/MSMD/mert_emb
"""
from __future__ import annotations
import argparse, json
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


@torch.no_grad()
def encode_piece(model, wav_path, chunk_sec=10.0, device='cuda'):
    """Run frozen MERT over audio in chunks; return (T_native, 768) at 75 Hz."""
    y, _ = librosa.load(str(wav_path), sr=MERT_SR, mono=True)
    chunk = int(chunk_sec * MERT_SR)
    outs = []
    for s in range(0, len(y), chunk):
        seg = y[s:s + chunk]
        if len(seg) < MERT_SR // 4:   # skip very short tails
            break
        t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(device)
        out = model(input_values=t).last_hidden_state[0]   # (T_chunk, 768)
        outs.append(out.cpu().float().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 768), dtype=np.float32)


def resample_emb(emb, src_fps, dst_fps):
    """Linear interpolation of (T_src, D) from src_fps to dst_fps."""
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
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--out',       default='data/MSMD/mert_emb')
    p.add_argument('--fps',       type=int, default=20)
    p.add_argument('--splits',    nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--mert_id',   default='m-a-p/MERT-v1-95M')
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading MERT ({a.mert_id}) on {device}...', flush=True)
    model = _load_model(a.mert_id, device)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = Path(a.processed)
    splits_data = json.load(open(proc / 'splits.json'))
    piece_ids = []
    for s in a.splits:
        piece_ids.extend(splits_data.get(s, []))
    piece_ids = list(dict.fromkeys(piece_ids))   # deduplicate, preserve order

    done = skip = fail = 0
    print(f'Encoding {len(piece_ids)} pieces at {a.fps} Hz...', flush=True)
    for i, pid in enumerate(piece_ids):
        out_path = out_dir / f'{pid}.npy'
        if out_path.exists():
            skip += 1
            continue
        wav = proc / pid / 'audio.wav'
        if not wav.exists():
            print(f'  SKIP {pid}: no audio.wav', flush=True)
            fail += 1
            continue
        try:
            emb = encode_piece(model, wav, device=device)        # (T_75, 768)
            if emb.shape[0] == 0:
                print(f'  SKIP {pid}: empty', flush=True); fail += 1; continue
            emb20 = resample_emb(emb, MERT_FPS, a.fps)          # (T_20, 768)
            np.save(out_path, emb20.astype(np.float16))
            done += 1
        except Exception as e:
            print(f'  FAIL {pid}: {e}', flush=True); fail += 1
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(piece_ids)}] done={done} skip={skip} fail={fail}', flush=True)

    print(f'Done. done={done} skip={skip} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
