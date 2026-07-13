"""Parse a f3_ensemble_decode smoke-test log and decide whether the
candidate decoder is worth a full 94-piece run. Exits 0 (PROCEED) if the
candidate's pct@0.5s on the smoke sample is within `min_delta` percentage
points of the baseline decoder (or better); exits 1 (SKIP) otherwise.

Used to gate expensive full runs behind SLURM job dependencies so they only
fire -- and only actually run -- when the smoke test already showed a real
signal, without a human needing to inspect the log first.

    python scripts/check_smoke_and_decide.py --log results/smoke_f6_graph-JOBID.log \
        --baseline original --candidate repeat_graph_snap --min_delta -1.0
"""
from __future__ import annotations
import argparse
import re


def parse_pct_05(log_text: str, decoder: str) -> float | None:
    """Find the '=== F3 (...) decoder=<decoder> on ... ===' block and pull
    its pct@0.5s value. Returns None if not found (job crashed / decoder
    name mismatch)."""
    pattern = re.compile(
        rf'decoder={re.escape(decoder)} on \S+ ===\n(?:.*\n)*?\s*pct@0\.5s\s*=\s*([\d.]+)%'
    )
    m = pattern.search(log_text)
    return float(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True)
    p.add_argument('--baseline', default='original')
    p.add_argument('--candidate', required=True)
    p.add_argument('--min_delta', type=float, default=-1.0,
                   help='candidate must be >= baseline + min_delta (percentage points) to PROCEED')
    a = p.parse_args()

    text = open(a.log).read()
    base_pct = parse_pct_05(text, a.baseline)
    cand_pct = parse_pct_05(text, a.candidate)

    if base_pct is None or cand_pct is None:
        print(f'DECISION: SKIP -- could not parse pct@0.5s for baseline={a.baseline} '
              f'({base_pct}) or candidate={a.candidate} ({cand_pct}) from {a.log}')
        raise SystemExit(1)

    delta = cand_pct - base_pct
    if delta >= a.min_delta:
        print(f'DECISION: PROCEED -- {a.candidate}={cand_pct:.1f}% vs {a.baseline}={base_pct:.1f}% '
              f'(delta={delta:+.1f}pp >= min_delta={a.min_delta:+.1f}pp)')
        raise SystemExit(0)
    else:
        print(f'DECISION: SKIP -- {a.candidate}={cand_pct:.1f}% vs {a.baseline}={base_pct:.1f}% '
              f'(delta={delta:+.1f}pp < min_delta={a.min_delta:+.1f}pp)')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
