#!/bin/bash
#SBATCH --job-name=c2-temporal
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/c2_temporal-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/c2_temporal-%j.log

# C2 -- causal temporal decode on the RELEASED cyolo_sb (79.9). No training.
# get_max_box takes a bare per-frame argmax over objectness with nothing linking
# consecutive frames; this replaces it with a filtered decision. Baseline to
# beat on `room`: 79.9.
#
#   usage: sbatch eval_c2_temporal_cpu.sh [lam] [fwd_px] [sigma_px]

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate

CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK

export C2_LAM=${1:-1.0} C2_FWD=${2:-6.0} C2_SIGMA=${3:-18.0}
CKPT=$CY/trained_models/cyolo_sb/best_model.pt
echo "checkpoint: $CKPT   lam=$C2_LAM fwd=$C2_FWD sigma=$C2_SIGMA"

echo ""; echo "########## msmd_rp / room ##########"
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_c2_temporal.py \
    --param_path "$CKPT" --test_dirs "$DATA/msmd_rp" \
    --split_files "$DATA/split_files/room_split.yaml" --only_onsets 2>&1 | tail -20

echo ""; echo "Job finished at $(date)"
