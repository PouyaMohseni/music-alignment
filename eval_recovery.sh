#!/bin/bash
#SBATCH --job-name=recover
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/recover-%j.log
# Attack the failure the trajectories actually show.
#
# 70.9% of our remaining error frames sit in runs of five or more consecutive
# onsets, the longest lasting 16.3 s. Against the raw baseline we cut the number
# of error runs from 289 to 126 -- the prior removed the jitter -- but the
# survivors got LONGER (p90 6 -> 9 onsets, max 45 -> 57). The smoother is both
# the cure for jitter and the reason a wrong commitment persists.
#
# Two independent fixes for that, neither adding a parameter to the model:
#
#  A. RE-ANCHOR. If the detector's own most confident box has sat far from the
#     tracked position for k onsets running, stop arguing with it and jump. One
#     frame of disagreement is noise; five in a row is being lost.
#
#  B. DISCOUNTED BEAM. `discount` was written for exactly this and then never
#     exposed or swept. It is a NO-OP at beam=1, so the untested cell is beam>1
#     WITH forgetting -- multiple hypotheses whose accumulated evidence has a
#     finite half-life, so fresh evidence can still overturn a wrong commit.
#     Plain wide beams were tested and degrade monotonically (84.7/84.3/84.3/
#     83.8/83.8 for width 1/4/8/16/32); the diagnosis was unbounded accumulation
#     and this is the direct test of that diagnosis.
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
REC=/scratch/pmohseni/omr/recover; mkdir -p "$REC"
export SEARCH_KIND=beam C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0
export CLUSTER_PX=0 C2_TOPK=100000 TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0 C2_CLASSES=0

run () { export REC_OUT="$REC/$1.npz" TRAJ_OUT="$REC/$1.traj.npz"
    echo ""; echo "##### $1  beam=$BEAM discount=$DISCOUNT reanchor=${REANCHOR_K}/${REANCHOR_PX}px"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |^Average accuracy|\[TRAJ\]|rror|Traceback"; }

export BEAM=1 DISCOUNT=1.0 REANCHOR_K=0 REANCHOR_PX=200
run control                       # must return 86.5 exactly

echo ""; echo "=== A. re-anchor on persistent disagreement ==="
for PX in 150 300; do for K in 2 3 5 8; do
    export REANCHOR_K=$K REANCHOR_PX=$PX; run "ra_k${K}_${PX}px"
done; done
export REANCHOR_K=0

echo ""; echo "=== B. beam with a finite memory ==="
for B in 4 8; do for D in 0.9 0.7 0.5 0.0; do
    export BEAM=$B DISCOUNT=$D; run "beam${B}_d${D}"
done; done
echo ""; echo "Job finished at $(date)"
