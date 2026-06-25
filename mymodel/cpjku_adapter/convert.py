"""Convert our MSMD processed format to the CPJKU audio_conditioned_unet format.

Produces per-piece files compatible with both our eval_official.py adapter
and their unmodified native train_model.py / eval_model.py:

    <cpjku_data>/score/<piece_id>.npz
        sheet      (H, W) uint8
        coords     (N, 3) float32: [y=H//2, x=strip_x, height=H//2]
                   Three columns required by their dataset.py:
                     true_position, height = result[:-1], result[-1]
        coord2onset  array([{note_idx: onset_idx, ...}], dtype=object)
        onset_frames (N,) int64

    <cpjku_data>/performance/<piece_id>.wav      ← our cpjku_adapter eval
    <cpjku_data>/performance/<piece_id>_1000.wav ← their native pipeline
                                                   (real_perf=True, tempo_factor=1000)
    <cpjku_data>/performance/<piece_id>.mid      ← always loaded for onset times

    python -m mymodel.cpjku_adapter.convert \
        --processed data/MSMD/processed \
        --out       data/MSMD/cpjku_fmt \
        --splits    train val test
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

    # ── Note coordinates (y=H//2, x=strip_x, height=H//2) ────────────────────
    # Three columns required by their dataset.py:
    #   true_position, height = result[:-1], result[-1]
    # Column layout: [y, x, height] where height drives the GT rectangle size.
    strip_x   = notes['strip_x'].astype(np.float32)    # (N,)
    onset_sec = notes['onset_sec'].astype(np.float64)  # (N,)
    N = len(strip_x)

    coords = np.stack([
        np.full(N, H // 2,   dtype=np.float32),   # y: vertical centre
        strip_x,                                    # x: horizontal position
        np.full(N, H // 2,   dtype=np.float32),   # height: GT rect height
    ], axis=1)   # (N, 3)

    # ── coord2onset: note_idx → onset_frame at fps ───────────────────────────
    onset_frames = np.round(onset_sec * fps).astype(np.int64)  # (N,)
    coords2onsets = {i: i for i in range(N)}                   # identity: each note → its own onset slot
    coord2onset = np.array([coords2onsets], dtype=object)

    npz_path = out_score / f'{pid}.npz'
    np.savez(npz_path, sheet=strip_arr, coords=coords,
             coord2onset=coord2onset, onset_frames=onset_frames)

    # ── Audio: symlink our audio.wav ─────────────────────────────────────────
    # <pid>.wav      — used by our cpjku_adapter eval code
    # <pid>_1000.wav — used by their native load_performance(real_perf=True, tempo_factor=1000)
    wav_src = piece_dir / 'audio.wav'
    for wav_name in (f'{pid}.wav', f'{pid}_1000.wav'):
        wav_dst = out_perf / wav_name
        if wav_src.exists() and not wav_dst.exists():
            os.symlink(wav_src.resolve(), wav_dst)

    # ── MIDI: symlink our score.midi as <pid>.mid ─────────────────────────────
    # Their load_performance always loads the MIDI to extract onset times, even
    # with real_perf=True.  They expect <pid>.mid (not .midi).
    mid_src = piece_dir / 'score.midi'
    mid_dst = out_perf / f'{pid}.mid'
    if mid_src.exists() and not mid_dst.exists():
        os.symlink(mid_src.resolve(), mid_dst)

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
