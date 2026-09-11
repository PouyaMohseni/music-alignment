#!/bin/bash
#SBATCH --job-name=resid
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/resid_%a-%A.log
# The scorer as a CORRECTION on top of the hand score, so no blend weight has
# to be chosen. The blend weight is the one constant no proxy could select:
# synthetic audio says trust the scorer completely, the real recordings say
# 0.6-0.7, and that disagreement cost the honest protocol several points.
# Decoding a residual checkpoint is argmax(s + h), i.e. blend 0.5.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/resid; mkdir -p "$M"
S=${SLURM_ARRAY_TASK_ID}
T="$M/resid_s$S.pt"
[ -f "$T" ] && { echo "present: $T"; exit 0; }
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 64 --seed "$S" --no_tempo --residual 2>&1 \
  | grep -E "RESIDUAL|BEST|wrote|rror|Traceback" | tail -5
[ -f "$T" ] || { echo "NO CHECKPOINT"; exit 1; }
