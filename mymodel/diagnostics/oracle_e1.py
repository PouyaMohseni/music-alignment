"""E1 — Oracle pianoroll alignment ceiling.

THE decisive, training-free diagnostic from REDESIGN.md. It asks:
  "If BOTH modalities were perfectly transcribed to pitch, does the 1-D strip +
   banded DTW framing actually align to sub-second accuracy?"

It builds, per piece, an oracle pitch representation on BOTH sides directly from
the MIDI ground truth we already store (noteheads.npz: midi_pitch, onset_sec,
midi_offset_sec, strip_x):
  - audio side : (T, 88) pianoroll over time frames (note active onset..offset)
  - score side : (N, 88) pianoroll over strip tiles (note active in every tile
                 whose 224-px receptive field contains its strip_x)
then cosine -> (T, N) -> existing dtw_backtrack -> existing alignment/henkel
metrics. No neural network, no training.

Interpretation:
  * error collapses toward sub-second  => pitch space + 1-D strip is alignable;
    the bottleneck is FEATURE DISCRIMINABILITY (RC2). Build the pitch-aware model.
  * error stays ~5 s even with perfect pitch => the global-matrix + DTW FRAMING
    (RC1) is the wall; no feature fix alone will help -> go to conditioned following.

Usage:
    python -m mymodel.diagnostics.oracle_e1 --split test
    python -m mymodel.diagnostics.oracle_e1 --split test --fps 20 --band 0.25 --limit 10
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

from ..shared.metrics import (alignment_metrics, dtw_backtrack,
                              henkel_metrics, retrieval_metrics)

MIDI_LOW = 21   # A0
N_PITCH = 88


def _audio_pianoroll(onset_sec, offset_sec, pitch, T, fps) -> np.ndarray:
    pr = np.zeros((T, N_PITCH), dtype=np.float32)
    for on, off, p in zip(onset_sec, offset_sec, pitch):
        k = int(p) - MIDI_LOW
        if k < 0 or k >= N_PITCH:
            continue
        a = max(0, int(round(on * fps)))
        b = min(T, int(round(off * fps)) + 1)
        if b <= a:
            b = min(T, a + 1)
        pr[a:b, k] = 1.0
    return pr


def _score_pianoroll(strip_x, pitch, tile_centers, tile_size) -> np.ndarray:
    N = len(tile_centers)
    pr = np.zeros((N, N_PITCH), dtype=np.float32)
    half = tile_size / 2.0
    for x, p in zip(strip_x, pitch):
        k = int(p) - MIDI_LOW
        if k < 0 or k >= N_PITCH:
            continue
        # every tile whose 224-px receptive field contains this notehead x
        active = np.abs(tile_centers - float(x)) <= half
        pr[active, k] = 1.0
    return pr


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return an @ bn.T                       # (T, N)


def eval_piece(pid, processed_root, fps, tile_size, tile_stride, band) -> dict | None:
    pdir = Path(processed_root) / pid
    ann = json.load(open(pdir / "annotations.json"))
    notes = np.load(pdir / "noteheads.npz")
    onset = notes["onset_sec"].astype(np.float64)
    offset = notes["midi_offset_sec"].astype(np.float64)
    pitch = notes["midi_pitch"].astype(np.int64)
    strip_x = notes["strip_x"].astype(np.float64)
    if len(onset) == 0:
        return None

    strip_w = int(ann["image"]["width_px"])
    dur = float(ann["audio"]["duration_sec"])
    px_per_sec = strip_w / dur

    T = int(np.ceil(dur * fps)) + 1
    N = max(1, (strip_w - tile_size) // tile_stride + 1)
    tile_centers = np.arange(N) * tile_stride + tile_size / 2.0

    audio_pr = _audio_pianoroll(onset, offset, pitch, T, fps)
    score_pr = _score_pianoroll(strip_x, pitch, tile_centers, tile_size)

    sim = _cosine_sim(audio_pr, score_pr)              # (T, N)
    path = dtw_backtrack(sim, band_radius_frac=band)   # (P, 2) of (t, n)

    pred_tile = np.zeros(T, dtype=np.int64)
    for t, n in path:
        pred_tile[t] = n
    pred_x_per_frame = tile_centers[pred_tile]

    frame = np.clip(np.round(onset * fps).astype(int), 0, T - 1)
    pred_at_onset = pred_x_per_frame[frame]

    m = alignment_metrics(
        pred_at_onset, strip_x, px_per_sec,
        beat_times_sec=ann.get("beat_times_sec") or None,
        bar_times_sec=ann.get("bar_times_sec") or None,
        gt_onset_sec=onset)
    m.update(retrieval_metrics(sim))
    m.update(henkel_metrics(pred_at_onset, strip_x))
    m["piece_id"] = pid
    m["sim_shape"] = list(sim.shape)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/MSMD/processed")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--fps", type=int, default=20, help="audio pianoroll frame rate")
    p.add_argument("--tile_size", type=int, default=224)
    p.add_argument("--tile_stride", type=int, default=56)
    p.add_argument("--band", type=float, default=0.25, help="DTW band radius frac")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out_dir", default="results/oracle_e1")
    args = p.parse_args()

    splits = json.load(open(Path(args.processed) / "splits.json"))
    ids = splits[args.split]
    if args.limit:
        ids = ids[: args.limit]

    out_root = Path(args.out_dir) / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"E1 oracle pianoroll | split={args.split} pieces={len(ids)} "
          f"fps={args.fps} band={args.band}", flush=True)

    rows = []
    with open(out_root / "per_piece.jsonl", "w") as f:
        for k, pid in enumerate(ids):
            try:
                m = eval_piece(pid, args.processed, args.fps,
                               args.tile_size, args.tile_stride, args.band)
            except Exception as e:
                m = {"piece_id": pid, "error": repr(e)}
            if m is None:
                continue
            rows.append(m)
            f.write(json.dumps(m) + "\n"); f.flush()
            if (k + 1) % 10 == 0 and "error" not in rows[-1]:
                print(f"  [{k+1}/{len(ids)}] mean_abs_err_sec="
                      f"{np.mean([r['mean_abs_err_sec'] for r in rows if 'error' not in r]):.3f}",
                      flush=True)

    good = [r for r in rows if "error" not in r]
    keys = [kk for kk in good[0] if kk.startswith(("mean_", "median_", "pct_", "recall_")) or kk == "n"]
    summ = {"n_pieces": len(good), "n_errors": len(rows) - len(good),
            "fps": args.fps, "band": args.band}
    for kk in keys:
        vals = np.asarray([r[kk] for r in good if isinstance(r.get(kk), (int, float))])
        if len(vals):
            summ[f"mean_{kk}"] = float(vals.mean())
            summ[f"median_{kk}"] = float(np.median(vals))
    with open(out_root / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
