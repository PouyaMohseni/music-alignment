"""Convert our MSMD processed format to the CPJKU audio_conditioned_unet format.

CPJKU expects:
    <cpjku_data>/
        score/<piece_id>.npz   → sheet (H, W) uint8, coords (N, 2), coord2onset
        performance/<piece_id>.wav  → synthesised audio (symlinked from our audio.wav)

Our strip.png (H_strip × W) is used as-is for `sheet`.
Note coords: (y=H//2, x=strip_x[i]) in strip pixel space.
coord2onset[0]: {note_idx → onset_frame_idx} at fps=20.

    python -m mymodel.cpjku_adapter.convert \
        --processed data/MSMD/processed \
        --out       data/MSMD/cpjku_fmt \
        --splits    test
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path

import numpy as np
from PIL import Image


def convert_piece(piece_dir: Path, out_score: Path, out_perf: Path, fps: int = 20):
    ann   = json.load(open(piece_dir / 'annotations.json'))
    notes = np.load(piece_dir / 'noteheads.npz')
    pid   = piece_dir.name

    # ── Score image ──────────────────────────────────────────────────────────
    strip_img = Image.open(piece_dir / 'strip.png').convert('L')
    strip_arr = np.array(strip_img, dtype=np.uint8)   # (H, W) 0=background 255=note
    H, W = strip_arr.shape

    # ── Note coordinates (y=H//2, x=strip_x) in strip pixel space ────────────
    strip_x  = notes['strip_x'].astype(np.float32)    # (N,)
    onset_sec = notes['onset_sec'].astype(np.float64)  # (N,)
    N = len(strip_x)

    coords = np.stack([
        np.full(N, H // 2, dtype=np.float32),
        strip_x
    ], axis=1)   # (N, 2): [y, x]

    # ── coord2onset: note_idx → onset_frame at fps ───────────────────────────
    onset_frames = np.round(onset_sec * fps).astype(np.int64)  # (N,)
    coords2onsets = {i: i for i in range(N)}                   # identity: each note → its own onset slot
    # They call: merge_onsets(cur_onsets, coords, coord2onset[0])
    # cur_onsets = midi_onset_frames; we pass onset_frames directly
    # We store coord2onset as a list-of-one-dict to match npz allow_pickle structure
    coord2onset = np.array([coords2onsets], dtype=object)

    npz_path = out_score / f'{pid}.npz'
    np.savez(npz_path, sheet=strip_arr, coords=coords,
             coord2onset=coord2onset, onset_frames=onset_frames)

    # ── Audio: symlink our audio.wav as <pid>.wav ─────────────────────────────
    wav_src = piece_dir / 'audio.wav'
    wav_dst = out_perf / f'{pid}.wav'
    if wav_src.exists() and not wav_dst.exists():
        os.symlink(wav_src.resolve(), wav_dst)

    return N


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--out',       default='data/MSMD/cpjku_fmt')
    p.add_argument('--splits',    nargs='+', default=['test'])
    p.add_argument('--fps',       type=int, default=20)
    a = p.parse_args()

    proc = Path(a.processed)
    out  = Path(a.out)
    out_score = out / 'score';      out_score.mkdir(parents=True, exist_ok=True)
    out_perf  = out / 'performance'; out_perf.mkdir(parents=True, exist_ok=True)

    splits = json.load(open(proc / 'splits.json'))
    piece_ids = []
    for s in a.splits:
        piece_ids.extend(splits.get(s, []))
    piece_ids = list(dict.fromkeys(piece_ids))

    print(f'Converting {len(piece_ids)} pieces to CPJKU format → {out}', flush=True)
    done = fail = 0
    for pid in piece_ids:
        piece_dir = proc / pid
        try:
            N = convert_piece(piece_dir, out_score, out_perf, fps=a.fps)
            done += 1
        except Exception as e:
            print(f'  FAIL {pid}: {e}', flush=True); fail += 1

    print(f'Done. converted={done} failed={fail}', flush=True)


if __name__ == '__main__':
    main()
