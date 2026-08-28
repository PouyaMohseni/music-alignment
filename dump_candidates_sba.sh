#!/bin/bash
#SBATCH --job-name=cand-sba
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cand_sba-%j.log
# cyolo_sb_a candidates, so the offline testbed can sweep over the stronger model
# too -- including its own candidate ceiling, which bounds what any readout can
# reach on it.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 DUMP_MAXK=256
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
OUT=/scratch/pmohseni/omr/cand_sba; mkdir -p "$OUT"
for TIER in room rp_synth; do
    export DUMP_OUT="$OUT/$TIER.npz"
    echo ""; echo "##### sb_a $TIER"
    python extensions/hooks/run_eval_dump.py \
        --param_path "$CY/trained_models/cyolo_sb_a/best_model.pt" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/${TIER}_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |\[DUMP\]|rror"
done
