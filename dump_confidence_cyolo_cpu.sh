#!/bin/bash
#SBATCH --job-name=dump-conf-cyolo
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dump_conf_cyolo-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/dump_conf_cyolo-%j.log

# Dump per-frame confidence + tracking error for released CYOLO checkpoints, so
# the calibration study can score continuous confidence against the two published
# hand-built heuristics (Brazier & Widmer EUSIPCO 2021 reliability factor; CODA
# silence break-mode).  See scripts/dump_confidence_cyolo.py for definitions.
#
#   sbatch dump_confidence_cyolo_cpu.sh [MODEL] [TIER]

set -uo pipefail
MODEL=${1:-cyolo_sb}
TIER=${2:-room}
echo "Job started on $(hostname) at $(date): model=$MODEL tier=$TIER"

module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate

REPO=/project/def-ichiro/pmohseni/music-alignment
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:${PYTHONPATH:-}
mkdir -p $REPO/results/calibration
cd /scratch/pmohseni/datasets/cyolo_score_following/cyolo_score_following

python $REPO/scripts/dump_confidence_cyolo.py \
    --model "$MODEL" --tier "$TIER" --num_workers 6 \
    --out "$REPO/results/calibration/${MODEL}_${TIER}_allframes.npz"

echo "Job finished at $(date)"
