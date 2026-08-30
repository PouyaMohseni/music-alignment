#!/bin/bash
#SBATCH --job-name=zsel
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/zsel-%j.log
# z_only wins the validation ranking at 95.60, so the protocol says run it on
# room. But the z-versus-no-z evidence is contradictory and probably noise:
#
#   controlled, same union data:  z_union 95.04  <  noz_union 95.28   (-0.24)
#   across dumps:                 z_only  95.60  >  ir_only   95.12   (+0.48)
#
# 0.5 points on 3728 frames is 19 frames. So this arm tests the validation
# winner honestly; it does NOT establish that z is what made it win. The
# noz_only control being trained alongside is what would settle that.
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
REC=/scratch/pmohseni/omr/zsel; mkdir -p "$REC"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { export REC_OUT="$REC/$1_$2.npz" SCORER_PATH=$M/$3.pt SCORER_BLEND=$4
    [ -f "$M/$3.pt" ] || { echo ""; echo "##### $1: no $3.pt yet, skipped"; return; }
    echo ""; echo "##### $1  tier=$2  $3 blend=$4"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/$2_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |^Average|rror"; }

run z_only    room z_only    0.7
run noz_union room noz_union 0.7
run noz_only  room noz_only  0.7
run z_only    do   z_only    0.7
echo ""; echo "Job finished at $(date)"
