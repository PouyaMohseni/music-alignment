#!/bin/bash
# Deferred resubmission of the N1 / N2 temporal-architecture experiments.
#
# Both were cancelled 2026-08-03 while the queue was fairshare-throttled, to
# stop them competing with N3 and the cyolo_sb reproduction. Their checkpoints
# are intact, so these RESUME rather than restart:
#   N1 long-context      24.1 pct@0.5s -- LSTM replaced, trained from scratch,
#                        only ~3h so far; the low score is undertraining, not a
#                        verdict on the idea.
#   N2 memory retrieval  86.6 pct@0.5s -- warm-started from B1a (89.2) and FELL
#                        2.6 points, so this one has a real negative signal;
#                        more hours may or may not recover it.
#
# Run when there is GPU headroom:  bash resubmit_n1_n2.sh
set -uo pipefail
cd "$(dirname "$0")"
for s in train_n1_long_context.sh train_n2_memory_retrieval.sh; do
    printf "%-34s -> " "$s"; sbatch --parsable "$s"
done
echo "Both resume from latest_model.pt; 3h chained runs, backfill-friendly."
