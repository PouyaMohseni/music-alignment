#!/bin/bash
#SBATCH --job-name=cross-tier
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cross_tier-%j.log
# Every decode hyperparameter we ship was chosen by sweeping on the ROOM tier.
# So run the same frozen checkpoint and the same decode on the two tiers it has
# never seen -- rp_synth (synthetic audio) and do (direct pickup) -- against the
# untouched argmax through the identical harness.
#
# If the gain holds on tiers that never informed a single choice, it is a
# mechanism. If it only exists on room, we tuned to the test set and the number
# is not what we think it is.
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
REC=/scratch/pmohseni/omr/xtier; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0
export CLUSTER_PX=0 C2_TOPK=100000 ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
export TIME_SIG_POW=0 TIME_REF=5

run () { export REC_OUT="$REC/$1_$2.npz"
    echo ""; echo "##### tier=$2  arm=$1  C2_CLASSES='${C2_CLASSES}' mu=$TIME_MU_POW"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/$2_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$"; }

for TIER in rp_synth do room; do
    export C2_CLASSES='' TIME_MU_POW=0;  run baseline "$TIER"   # untouched argmax
    export C2_CLASSES='0' TIME_MU_POW=0; run decode   "$TIER"   # uncapped temporal filter
    export C2_CLASSES='0' TIME_MU_POW=1; run timeaware "$TIER"  # + time-aware mean
done
echo ""; echo "Job finished at $(date)"
