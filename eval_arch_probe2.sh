#!/bin/bash
#SBATCH --job-name=arch2
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/arch2-%j.log
# 1. The ceiling, rerun. The first attempt recorded 2 frames out of 4149: a
#    batch holds 32 items and they are usually the SAME piece, so the slice of
#    newly-appended frame_diffs is the whole batch's, not the item's, and the
#    len(new)==1 guard threw away everything else. Now indexed per item.
# 2. ref sensitivity for the winning time-aware prior (mu_pow=1, sig_pow=0,
#    86.5). ref=5 was the MEASURED median onset gap on the test set, which
#    makes it a test-derived constant -- so the question is whether the curve is
#    flat around it (a mechanism) or a spike (a fit).
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
REC=/scratch/pmohseni/omr/arch2; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0
export CLUSTER_PX=0 C2_TOPK=100000 ANCHOR=start WINDOW=0 Z_MASK=none
export ORACLE=0 TIME_MU_POW=0 TIME_SIG_POW=0 TIME_REF=5

run () { export REC_OUT="$REC/$1_room.npz" ORACLE_OUT="$REC/$1_cand.npz"
    echo ""; echo "##### $1  ORACLE=$ORACLE mu=$TIME_MU_POW sig=$TIME_SIG_POW ref=$TIME_REF"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$"; }

echo "=== A. candidate ceiling, on the shipped decode ==="
export ORACLE=1; run oracle_ctrl
echo ""; echo "=== B. candidate ceiling, on the time-aware decode ==="
export TIME_MU_POW=1; run oracle_time; export ORACLE=0

echo ""; echo "=== C. ref sensitivity for mu_pow=1, sig_pow=0 ==="
for R in 3 4 6 7 9; do export TIME_REF=$R; run mu1_r$R; done
