"""Sweep inference over every piece in a split, aggregate metrics.

    python -m mymodel.v1_baseline.eval \
        --checkpoint results/v1_baseline/checkpoint_001000.pt \
        --split test

Writes results/v1_baseline/eval/<split>/{per_piece.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pathlib import Path as _P

from .infer import align_piece


def _aggregate(per_piece: list[dict]) -> dict:
    if not per_piece:
        return {"n_pieces": 0}
    keys = [k for k in per_piece[0]
            if k.startswith(("mean_", "median_", "pct_within_")) or k == "n"]
    agg = {"n_pieces": len(per_piece)}
    for k in keys:
        vals = np.asarray([m[k] for m in per_piece if isinstance(m.get(k), (int, float))])
        if not len(vals):
            continue
        agg[f"mean_{k}"]   = float(vals.mean())
        agg[f"median_{k}"] = float(np.median(vals))
    return agg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--processed", default="data/MSMD/processed")
    p.add_argument("--config", default="configs/v1_baseline.yaml")
    p.add_argument("--out_dir", default="results/v1_baseline/eval")
    p.add_argument("--chunk_sec", type=float, default=20.0)
    p.add_argument("--band_radius_frac", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="optional: only run the first N pieces (debug)")
    args = p.parse_args()

    splits = json.load(open(_P(args.processed) / "splits.json"))
    pieces = splits[args.split]
    if args.limit:
        pieces = pieces[: args.limit]

    out_root = Path(args.out_dir) / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    per_piece_path = out_root / "per_piece.jsonl"
    summary_path = out_root / "summary.json"

    print(f"eval split={args.split}  n_pieces={len(pieces)}")
    rows: list[dict] = []
    with open(per_piece_path, "w") as f:
        for i, pid in enumerate(pieces):
            try:
                m = align_piece(
                    piece_id=pid,
                    checkpoint_path=args.checkpoint,
                    processed_root=args.processed,
                    config_path=args.config,
                    out_dir=str(out_root / "pieces"),
                    chunk_sec=args.chunk_sec,
                    band_radius_frac=args.band_radius_frac,
                    device=args.device,
                )
            except Exception as e:
                m = {"piece_id": pid, "error": repr(e)}
            rows.append(m)
            f.write(json.dumps(m) + "\n")
            f.flush()
            if (i + 1) % 5 == 0:
                print(f"  [{i + 1}/{len(pieces)}] running mean_abs_err_sec="
                      f"{np.mean([r.get('mean_abs_err_sec', np.nan) for r in rows]):.3f}")

    summary = _aggregate([r for r in rows if "error" not in r])
    summary["split"] = args.split
    summary["checkpoint"] = args.checkpoint
    summary["n_errors"] = sum(1 for r in rows if "error" in r)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
