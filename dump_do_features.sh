#!/bin/bash
#SBATCH --job-name=do-feat
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/do_feat-%j.log
# `do` with backbone features, so it can rank the feature-based models.
#
# `do` is the only proxy with real signal against room (Spearman +0.60, where
# synthetic validation is +0.30 and rp_synth +0.15) and it is real audio 3.6
# points from room in difficulty. But cand_test was dumped without features, so
# it cannot rank feat_wide -- which is the model that reached 94.0 on room and
# is the one number I currently cannot defend.
#
# Piece identity overlaps with room, so this is not a clean holdout and any
# selection made on it must say so. The acoustic condition genuinely differs,
# which is the axis synthetic validation fails on.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 DUMP_MAXK=256 DUMP_FEATK=128
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK IR_PATH
O=/scratch/pmohseni/omr/candf; mkdir -p "$O"
for TIER in do rp_synth; do
    T="$O/$TIER.npz"
    if [ -f "$T" ] && [ "$(stat -c %s "$T")" -gt 1000000 ]; then
        echo "##### $TIER present"; continue; fi
    export DUMP_OUT="$T"
    echo ""; echo "##### $TIER with features"
    python extensions/hooks/run_eval_dump.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/${TIER}_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |\[DUMP\]|\[FEAT\]|rror|Traceback"
done
echo ""; echo "Job finished at $(date)"
