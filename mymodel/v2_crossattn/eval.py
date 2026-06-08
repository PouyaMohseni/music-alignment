"""v2 eval sweep over a split.

    python -m mymodel.v2_crossattn.eval \
        --checkpoint results/v2_nce/checkpoint_004000.pt \
        --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .infer import align_piece

PROCESSED = "data/MSMD/processed"


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n_pieces": 0}
    keys = [k for k in rows[0] if k.startswith(("mean_", "median_", "pct_within_", "recall_")) or k == "n"]
    agg = {"n_pieces": len(rows)}
    for k in keys:
        vals = np.asarray([r[k] for r in rows if isinstance(r.get(k), (int, float))])
        if not len(vals): continue
        agg[f"mean_{k}"] = float(vals.mean())
        agg[f"median_{k}"] = float(np.median(vals))
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--processed", default=PROCESSED)
    p.add_argument("--config", default="configs/v2_crossattn.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint).parent
    out_dir = args.out_dir or str(ckpt_dir / "eval")

    splits = json.load(open(Path(args.processed) / "splits.json"))
    pieces = splits[args.split]
    if args.limit:
        pieces = pieces[: args.limit]

    print(f"eval split={args.split}  n_pieces={len(pieces)}  checkpoint={args.checkpoint}")
    out_root = Path(out_dir) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    with open(out_root / "per_piece.jsonl", "w") as f:
        for i, pid in enumerate(pieces):
            try:
                m = align_piece(pid, args.checkpoint,
                                processed_root=args.processed,
                                config_path=args.config,
                                out_dir=str(out_root / "pieces"),
                                device=args.device)
            except Exception as e:
                m = {"piece_id": pid, "error": repr(e)}
                errors.append(m)
            rows.append(m)
            f.write(json.dumps(m) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                good = [r for r in rows if "error" not in r]
                if good:
                    print(f"  [{i+1}/{len(pieces)}] running mean_abs_err_sec="
                          f"{np.mean([r.get('mean_abs_err_sec', np.nan) for r in good]):.3f}")

    summary = _aggregate([r for r in rows if "error" not in r])
    summary["split"] = args.split
    summary["checkpoint"] = args.checkpoint
    summary["n_errors"] = len(errors)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
