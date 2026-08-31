#!/bin/bash
#SBATCH --job-name=eval-scorer
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_scorer-%j.log
# The learned selector on the real test tiers. It has never seen room, and
# unlike every constant in the shipped prior it was not chosen there either.
#
# blend=0 runs the hand-tuned prior through the SAME decoder object, so the
# control is the identical code path with the learned term switched off, not a
# different script that happens to agree.
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
REC=/scratch/pmohseni/omr/scoreval; mkdir -p "$REC"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5
export CLUSTER_PX=0 ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { local tag=$1 tier=$2
    export REC_OUT="$REC/${tag}_${tier}.npz"
    echo ""; echo "##### $tag  tier=$tier  scorer=$(basename "$SCORER_PATH") blend=$SCORER_BLEND"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/${tier}_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$"; }

# control first: same decoder object, learned term off -> must return 86.5
export SCORER_PATH=$M/base.pt SCORER_BLEND=0.0; run control room

for V in base noabs nonoise strongnoise big; do
    [ -f "$M/$V.pt" ] || { echo "##### $V: no checkpoint, skipped"; continue; }
    export SCORER_PATH=$M/$V.pt SCORER_BLEND=1.0
    run "$V" room
done

# blend sweep on whichever variant is best is done in a follow-up; here just
# bracket the mixture on the default so we can see the shape
for B in 0.75 0.5 0.25; do
    export SCORER_PATH=$M/base.pt SCORER_BLEND=$B
    run "base_b$B" room
done

# and the tiers the selector has never seen either
export SCORER_PATH=$M/base.pt SCORER_BLEND=1.0
run base rp_synth
run base do
echo ""; echo "Job finished at $(date)"
