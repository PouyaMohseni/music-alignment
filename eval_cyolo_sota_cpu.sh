#!/bin/bash
#SBATCH --job-name=eval-cyolo-sota
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_cyolo_sota-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_cyolo_sota-%j.log

# Reproduce the actual SOTA under our own control: CPJKU's RELEASED CYOLO
# checkpoints (Henkel & Widmer 2021, Frontiers), which report 86.1% synthetic
# and 70.6% on real MSMD-Rec at pct@0.5s -- versus 41.8% for our best model on
# real audio, and versus the 12.5% their paper reports for the conditional
# U-Net family this project is built on.
#
# No training required: the repo ships trained weights (cyolo, cyolo_sb,
# cyolo_sb_a). Training would only be needed to MODIFY the architecture.
#
# Their msmd_rp tier carries three conditions selected by split file:
#   rp_synth  synthetic audio of the real-performance pieces  <- matched control
#   do        direct out
#   room      room microphone
# Evaluating all three on the same 16 pieces isolates acoustic domain shift
# from piece difficulty -- the confound our own tier comparison had.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import torch,madmom,librosa" || { echo "FATAL: venv_cyolo incomplete"; exit 1; }

CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export PYTHONPATH=$CY:${PYTHONPATH:-}
cd "$CY/cyolo_score_following"

MODEL=${1:-cyolo_sb_a}
# CYOLO_CKPT lets this evaluate OUR OWN trained checkpoints, not just the
# released ones under trained_models/.  Needed to read an interim number off the
# IR-augmented reproduction while it is still training: the run is 50 epochs at
# ~10 epochs per 24h allocation, so waiting for convergence is ~5 days, and the
# whole point of that run is to learn whether IR transfers to a detection model
# at all.
CKPT=${CYOLO_CKPT:-$CY/trained_models/$MODEL/best_model.pt}
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }
echo "model: $MODEL  ($CKPT)"

# CYOLO_SPLITS selects which evaluations to run. The synthetic msmd_test set is
# 94 pieces and on CPU it alone can exceed an 8h allocation -- job 669026 timed
# out inside it without ever reaching `room`, which was the only number wanted.
# Default keeps the original full behaviour.
CYOLO_SPLITS=${CYOLO_SPLITS:-synth rp_synth do room}

if [[ " $CYOLO_SPLITS " == *" synth "* ]]; then
  echo ""
  echo "########## SYNTHETIC msmd_test ##########"
  python eval.py --param_path "$CKPT" --test_dirs "$DATA/msmd_test" --only_onsets 2>&1 | tail -25
fi

for SPLIT in $CYOLO_SPLITS; do
  [ "$SPLIT" = "synth" ] && continue
  echo ""
  echo "########## msmd_rp / $SPLIT ##########"
  python eval.py --param_path "$CKPT" --test_dirs "$DATA/msmd_rp" \
      --split_files "$DATA/split_files/${SPLIT}_split.yaml" --only_onsets 2>&1 | tail -25
done

echo ""
echo "Job finished at $(date)"
