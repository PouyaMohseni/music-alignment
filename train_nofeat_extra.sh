#!/bin/bash
#SBATCH --job-name=trnofeat
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/trnofeat_%a-%A.log
# The shipped recipe on another detector, MINUS the 128 backbone numbers.
#
#   sbatch --array=0-2 train_nofeat_extra.sh cyolo_sb_a|cyolo
#
# On cyolo_sb the image features are worth nothing on room (featureless 90.89,
# DINOv2 90.89, cyolo features 90.6-91.5) while being worth +3.4 on synthetic
# validation. That was measured on ONE detector. 94.38 was trained WITH the
# features, so the honest statement today is "unmeasured on +A". This measures
# it: same command as train_extra_scorers.sh with --use_feat removed, so nf is
# still 33 hand features and only the backbone pathway (fenc, 128->64) is gone.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CELL=${1:?usage: train_nofeat_extra.sh cyolo_sb_a|cyolo}
S=${SLURM_ARRAY_TASK_ID}
F=/scratch/pmohseni/omr/cand_$CELL
[ -d "$F" ] || { echo "no dump $F"; exit 1; }
M=/scratch/pmohseni/omr/scorer/extra; mkdir -p "$M"
T="$M/${CELL}_nofeat_s$S.pt"
[ -f "$T" ] && { echo "present: $T"; exit 0; }
echo "##### $CELL nofeat seed=$S  dumps=$F"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --featproj 64 --seed "$S" --no_tempo 2>&1 | grep -E "BEST|wrote|rror|Traceback" | tail -5
[ -f "$T" ] || { echo "NO CHECKPOINT"; exit 1; }
