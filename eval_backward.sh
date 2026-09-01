#!/bin/bash
#SBATCH --job-name=backward
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/backward-%j.log
# Price a backward jump differently from a forward one.
#
# The prior floors at jump_logp = -6 once a displacement is far from the mean,
# and that floor was SYMMETRIC: a 400 px leap backwards cost exactly what a
# 400 px leap forwards cost. Measured on room:
#
#   ground truth steps backwards on   0.73% of onsets
#   our decoder steps backwards on    4.50%          -- six times too often
#
# and those steps are its worst: 35.7% correct in the 50-200 px band against
# 88.3% for a normal forward step. Not all of them are wrong, though -- 56.4%
# of the >200 px backward steps land correctly, and those look like recoveries
# from an earlier wrong commit. So this is a trade, not a free win, and the
# sweep is to find where it turns.
#
# C2_BACK unset reproduces the shipped prior bit-identically.
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
M=/scratch/pmohseni/omr/scorer
REC=/scratch/pmohseni/omr/backward; mkdir -p "$REC"
export C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { export REC_OUT="$REC/$1.npz" TRAJ_OUT="$REC/$1.traj.npz"
    echo ""; echo "##### $1  back=${C2_BACK:-symmetric}  kind=$SEARCH_KIND blend=${SCORER_BLEND:-n/a}"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |^Average|rror"; }

echo "=== A. hand decoder alone, so the effect is not mixed with the selector ==="
export SEARCH_KIND=beam BEAM=1 C2_CLASSES=0 C2_TOPK=100000
unset C2_BACK; run hand_ctrl
for B in -8 -10 -14 -20 -40; do export C2_BACK=$B; run "hand_b$B"; done
unset C2_BACK

echo ""; echo "=== B. with the shipped selector at blend 0.7 ==="
export SEARCH_KIND=scorer C2_TOPK=256 SCORER_PATH=$M/ir_only.pt SCORER_BLEND=0.7
run sel_ctrl
for B in -10 -14 -20; do export C2_BACK=$B; run "sel_b$B"; done
echo ""; echo "Job finished at $(date)"
