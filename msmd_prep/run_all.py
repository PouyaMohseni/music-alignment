"""Run all six MSMD-prep stages end-to-end.

    python -m msmd_prep.run_all \
        --raw     data/MSMD/msmd_aug_v1-1_no-audio \
        --splits  data/MSMD/msmd/msmd/splits/all_split.yaml \
        --out     data/MSMD/processed \
        [--sf2 path/to/piano.sf2]      # if omitted, audio is NOT synthesised
        [--all-performances]           # process all tempo/soundfont variants
        [--limit N]                    # for smoke tests
        [--jobs N]                     # parallel pieces (default cpu_count-2)

Default (no --all-performances):
    <out>/<piece_id>/strip.png
    <out>/<piece_id>/score.midi
    <out>/<piece_id>/annotations.json
    <out>/<piece_id>/noteheads.npz
    <out>/<piece_id>/audio.wav

With --all-performances:
    <out>/_strips/<piece_id>.png       (shared strip, symlinked per performance)
    <out>/<piece_id>__<perf_id>/score.midi
    <out>/<piece_id>__<perf_id>/annotations.json
    <out>/<piece_id>__<perf_id>/noteheads.npz
    <out>/<piece_id>__<perf_id>/audio.wav

Both modes produce:
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


def _worker(piece_dir: str, out_dir: str, sf2_path: str | None,
            all_performances: bool) -> dict:
    pid = os.path.basename(piece_dir.rstrip("/"))
    try:
        results = build_piece(piece_dir, out_dir, all_performances=all_performances)
        ok, failed = [], []
        for r in results:
            if "error" in r:
                failed.append(r)
            else:
                if sf2_path:
                    perf_dir = os.path.dirname(r["annotations"]) if all_performances else out_dir
                    try:
                        synthesize_piece(perf_dir, sf2_path=sf2_path)
                    except Exception as e:
                        r["error"] = repr(e)
                        failed.append(r)
                        continue
                ok.append(r)
        return {"ok": True, "piece_id": pid, "results": ok, "failures": failed}
    except Exception as e:
        return {"ok": False, "piece_id": pid, "reason": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(), "results": [], "failures": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="MSMD no-audio root")
    ap.add_argument("--splits", required=True, help="MSMD split yaml")
    ap.add_argument("--out", required=True, help="processed-dataset output dir")
    ap.add_argument("--sf2", default=None, help="SoundFont .sf2 path (enables Stage 2)")
    ap.add_argument("--all-performances", action="store_true",
                    help="process all tempo/soundfont variants (~13x more data)")
    ap.add_argument("--limit", type=int, default=None, help="cap to first N pieces (debug)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    raw = Path(args.raw).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    splits_yaml = yaml.safe_load(open(args.splits))
    splits = {("val" if k == "valid" else k): v for k, v in splits_yaml.items()}
    in_split = {pid: split for split, ids in splits.items() for pid in ids}

    piece_dirs = [raw / pid for pid in sorted(in_split) if (raw / pid).is_dir()]
    if args.limit:
        piece_dirs = piece_dirs[: args.limit]
    print(f"pipeline: raw={raw}  out={out}  pieces={len(piece_dirs)}  jobs={args.jobs}  "
          f"audio_synth={'yes' if args.sf2 else 'no'}  "
          f"all_performances={args.all_performances}", flush=True)

    exclusions: list[dict] = []
    # successes keyed by perf-level dir name for manifest
    success_dirs: list[str] = []
    # map dir_name -> piece_id for split lookup
    dir_to_piece: dict[str, str] = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_worker, str(pd), str(out / pd.name),
                          args.sf2, args.all_performances): pd
                for pd in piece_dirs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if not r["ok"]:
                exclusions.append({"piece_id": r["piece_id"], "reason": r["reason"]})
                print(f"  FAIL {r['piece_id']}: {r['reason']}", flush=True)
            else:
                for res in r["results"]:
                    if args.all_performances:
                        pid = res["piece_id"]
                        perf_id = res["performance_id"]
                        perf_suffix = perf_id[len(pid) + 1:] if perf_id.startswith(pid + "_") else perf_id
                        dir_name = f"{pid}__{perf_suffix}"
                    else:
                        dir_name = res["piece_id"]
                    success_dirs.append(dir_name)
                    dir_to_piece[dir_name] = res["piece_id"]
                for fail in r["failures"]:
                    exclusions.append(fail)
                    print(f"  FAIL {fail.get('performance_id','?')}: {fail.get('error','?')}",
                          flush=True)
            if done % 25 == 0:
                print(f"[{done}/{len(piece_dirs)}] ok={len(success_dirs)} "
                      f"fail={len(exclusions)} elapsed={time.time() - t0:.1f}s", flush=True)

    # Build splits using dir names (performance-level); map back through piece_id
    final_splits: dict[str, list[str]] = {k: [] for k in splits}
    for dir_name in success_dirs:
        piece_id = dir_to_piece[dir_name]
        for split_name, ids in splits.items():
            if piece_id in ids:
                final_splits[split_name].append(dir_name)
    final_splits = {k: sorted(v) for k, v in final_splits.items()}

    build_manifest(str(out), splits=final_splits)
    with open(out / "exclusions.json", "w") as f:
        json.dump(exclusions, f, indent=2)

    print(f"\nDONE in {time.time() - t0:.1f}s")
    print(f"  ok (performances) = {len(success_dirs)}")
    print(f"  failed            = {len(exclusions)}")
    print(f"  train             = {len(final_splits.get('train', []))}")
    print(f"  val               = {len(final_splits.get('val', []))}")
    print(f"  test              = {len(final_splits.get('test', []))}")


if __name__ == "__main__":
    main()
