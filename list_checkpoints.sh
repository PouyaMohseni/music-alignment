#!/usr/bin/env bash
# List all checkpoints per results dir, and the latest (highest-step) one each.
set -euo pipefail

RESULTS="${1:-results}"

echo "=================================================================="
printf "%-16s %-10s %s\n" "MODEL" "#CKPTS" "LATEST CHECKPOINT"
echo "=================================================================="

for d in "$RESULTS"/*/; do
    name=$(basename "$d")
    # collect checkpoint files, sorted by the zero-padded step in the name
    mapfile -t ckpts < <(ls "$d"checkpoint_*.pt 2>/dev/null | sort)
    n=${#ckpts[@]}
    if [[ $n -eq 0 ]]; then
        printf "%-16s %-10s %s\n" "$name" "0" "(none)"
    else
        latest=$(basename "${ckpts[$((n-1))]}")
        printf "%-16s %-10s %s\n" "$name" "$n" "$latest"
    fi
done

echo
echo "=================================================================="
echo "All checkpoints per model:"
echo "=================================================================="
for d in "$RESULTS"/*/; do
    name=$(basename "$d")
    mapfile -t ckpts < <(ls "$d"checkpoint_*.pt 2>/dev/null | sort)
    [[ ${#ckpts[@]} -eq 0 ]] && continue
    echo "  $name:"
    for c in "${ckpts[@]}"; do
        echo "    $(basename "$c")"
    done
done
