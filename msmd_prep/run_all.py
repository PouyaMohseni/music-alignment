"""Run all six MSMD-prep stages end-to-end.

    python -m msmd_prep.run_all \
        --raw     data/MSMD/msmd_aug_v1-1_no-audio \
        --splits  data/MSMD/msmd/splits/all_split.yaml \
        --out     data/MSMD/processed \
        [--sf2 path/to/piano.sf2]   # if omitted, audio is NOT synthesised
        [--limit N]                  # for smoke tests
        [--jobs N]                   # parallel pieces (default cpu_count-2)

Produces:
    <out>/<piece_id>/strip.png
    <out>/<piece_id>/score.midi
    <out>/<piece_id>/annotations.json
    <out>/<piece_id>/noteheads.npz
    <out>/<piece_id>/audio.wav        (if --sf2 given)
    <out>/manifest.jsonl
    <out>/splits.json
    <out>/exclusions.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from .manifest import build_manifest
from .piece import build_piece
from .synth import synthesize_piece


def _worker(piece_dir: str, out_dir: str, sf2_path: str | None) -> dict:
    try:
        info = build_piece(piece_dir, out_dir)
        if sf2_path:
            info["audio"] = synthesize_piece(out_dir, sf2_path=sf2_path)
        return {"ok": True, "piece_id": info["piece_id"], "n_notes": info["notehead_count"]}
    except Exception as e:
        pid = os.path.basename(piece_dir.rstrip("/"))
        return {"ok": False, "piece_id": pid, "reason": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="MSMD no-audio root")
    ap.add_argument("--splits", required=True, help="MSMD split yaml")
    ap.add_argument("--out", required=True, help="processed-dataset output dir")
    ap.add_argument("--sf2", default=None, help="SoundFont .sf2 path (enables Stage 2)")
    ap.add_argument("--limit", type=int, default=None, help="cap to first N pieces (debug)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    raw = Path(args.raw).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    splits_yaml = yaml.safe_load(open(args.splits))
    # Normalise key name 'valid' -> 'val' to match README spec
    splits = {("val" if k == "valid" else k): v for k, v in splits_yaml.items()}
    in_split = {pid: split for split, ids in splits.items() for pid in ids}

    piece_dirs = [raw / pid for pid in sorted(in_split) if (raw / pid).is_dir()]
    if args.limit:
        piece_dirs = piece_dirs[: args.limit]
    print(f"pipeline: raw={raw}  out={out}  pieces={len(piece_dirs)}  jobs={args.jobs}  "
          f"audio_synth={'yes' if args.sf2 else 'no'}", flush=True)

    exclusions: list[dict] = []
    successes: list[str] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_worker, str(pd), str(out / pd.name), args.sf2): pd
                for pd in piece_dirs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r["ok"]:
                successes.append(r["piece_id"])
            else:
                exclusions.append({"piece_id": r["piece_id"], "reason": r["reason"]})
                print(f"  FAIL {r['piece_id']}: {r['reason']}", flush=True)
            if done % 25 == 0:
                print(f"[{done}/{len(piece_dirs)}] ok={len(successes)} "
                      f"fail={len(exclusions)} elapsed={time.time() - t0:.1f}s",
                      flush=True)

    # Restrict splits to successes for manifest cleanliness
    final_splits = {k: sorted(set(v) & set(successes)) for k, v in splits.items()}
    build_manifest(str(out), splits=final_splits)
    with open(out / "exclusions.json", "w") as f:
        json.dump(exclusions, f, indent=2)

    print(f"\nDONE in {time.time() - t0:.1f}s")
    print(f"  ok      = {len(successes)}")
    print(f"  failed  = {len(exclusions)}")
    print(f"  train   = {len(final_splits.get('train', []))}")
    print(f"  val     = {len(final_splits.get('val', []))}")
    print(f"  test    = {len(final_splits.get('test', []))}")


if __name__ == "__main__":
    main()
