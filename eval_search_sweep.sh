#!/bin/bash
#SBATCH --job-name=search-sweep
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/search_sweep-%j.log
# Beam width sweep + banded Viterbi, all against the released cyolo_sb.
# BEAM=1 reproduces C2 exactly, so the sweep is a controlled test of whether
# GREEDY COMMITMENT is what costs us -- same scoring function throughout, only
# the search changes.
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

CKPT=$CY/trained_models/cyolo_sb/best_model.pt
REC=/scratch/pmohseni/omr/search
mkdir -p "$REC"

run () {   # $1 = tag
    export REC_OUT="$REC/$1_room.npz"
    echo ""; echo "########## $1 ##########"
    python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_search.py \
        --param_path "$CKPT" --test_dirs "$DATA/msmd_rp" \
        --split_files "$DATA/split_files/room_split.yaml" --only_onsets 2>&1 \
        | grep -vE "it/s\]|it\]" | tail -12
}

export SEARCH_KIND=beam C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0
for B in 1 4 8 16 32; do export BEAM=$B; run "beam${B}"; done

export SEARCH_KIND=viterbi
for BAND in 400 800; do export VIT_BAND=$BAND; run "viterbi${BAND}"; done

echo ""; echo "Job finished at $(date)"
