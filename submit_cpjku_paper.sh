#!/bin/bash
# Submit CB_TA training then eval as a dependent chain.
# Usage:
#   bash submit_cpjku_paper.sh            # train from scratch + auto-eval
#   bash submit_cpjku_paper.sh eval-only  # eval only (auto-discovers latest model)
#
# The eval job waits for training to complete successfully (afterok).
# If training fails the eval job is automatically cancelled by SLURM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "eval-only" ]; then
    echo "Submitting eval only (no training dependency)..."
    EVAL_OUT=$(sbatch "$SCRIPT_DIR/eval_cpjku_paper_test.sh")
    EVAL_JOB=$(echo "$EVAL_OUT" | grep -oP '\d+$')
    echo "  Eval job: $EVAL_JOB"
    echo "Monitor:  squeue -u $USER"
    exit 0
fi

echo "Submitting CB_TA training..."
TRAIN_OUT=$(sbatch "$SCRIPT_DIR/train_cpjku_paper_CB_TA.sh")
TRAIN_JOB=$(echo "$TRAIN_OUT" | grep -oP '\d+$')
echo "  Training job: $TRAIN_JOB"

echo "Submitting eval (depends on $TRAIN_JOB finishing successfully)..."
EVAL_OUT=$(sbatch --dependency=afterok:"$TRAIN_JOB" "$SCRIPT_DIR/eval_cpjku_paper_test.sh")
EVAL_JOB=$(echo "$EVAL_OUT" | grep -oP '\d+$')
echo "  Eval job:     $EVAL_JOB"

echo ""
echo "Chain: train $TRAIN_JOB → eval $EVAL_JOB"
echo "Monitor: squeue -u $USER"
echo "Cancel both: scancel $TRAIN_JOB $EVAL_JOB"
