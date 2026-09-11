#!/bin/bash
#SBATCH --job-name=trextra
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/trextra_%a-%A.log
# The shipped recipe (nbrp64) on other inputs, seed = array index.
#
#   sbatch --array=0-2 train_extra_scorers.sh cyolo|cyolo_sb_a|nohist
#
#   cyolo, cyolo_sb_a  retrained on that detector's own candidates
#   nohist             cyolo_sb candidates, every history feature removed
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CELL=${1:?usage: train_extra_scorers.sh cyolo|cyolo_sb_a|nohist}
S=${SLURM_ARRAY_TASK_ID}
PY=extensions/analysis/train_cand_scorer.py
case $CELL in
  cyolo|cyolo_sb_a) F=/scratch/pmohseni/omr/cand_$CELL ;;
  nohist) F=/scratch/pmohseni/omr/candf; PY=extensions/analysis/train_nohist.py ;;
  *) echo "unknown cell $CELL"; exit 1 ;;
esac
M=/scratch/pmohseni/omr/scorer/extra; mkdir -p "$M"
T="$M/${CELL}_s$S.pt"
[ -f "$T" ] && { echo "present: $T"; exit 0; }
echo "##### $CELL seed=$S  dumps=$F"
python "$PY" --out "$T" --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 64 --seed "$S" --no_tempo 2>&1 | grep -E "nohist|BEST|wrote|rror|Traceback" | tail -5
[ -f "$T" ] || { echo "NO CHECKPOINT"; exit 1; }
