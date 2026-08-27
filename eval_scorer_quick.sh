#!/bin/bash
#SBATCH --job-name=scorer-q
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/scorer_quick-%j.log
# Just the decisive pair, sized to backfill: does the selector fit on the
# training split beat the hand-tuned prior on ROOM? Two arms through the SAME
# decoder object, learned term off then on, so the control is the identical
# code path rather than a different script that happens to agree.
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
REC=/scratch/pmohseni/omr/scoreq; mkdir -p "$REC"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5
export CLUSTER_PX=0 ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
export SCORER_PATH=/scratch/pmohseni/omr/scorer/base.pt

run () { export REC_OUT="$REC/$1_room.npz" SCORER_BLEND=$2
    echo ""; echo "##### $1  blend=$2"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$"; }

run control 0.0     # hand-tuned prior, must return 86.5
run learned 1.0     # the selector
echo ""; echo "Job finished at $(date)"
