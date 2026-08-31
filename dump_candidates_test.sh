#!/bin/bash
#SBATCH --job-name=cand-test
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cand_test-%j.log
# Candidates for the three test tiers, so an OFFLINE decoder can be checked
# against the harness's own number before it is trusted. Sweeping decoders
# through eval.py costs eight minutes each; over a dumped candidate set it costs
# milliseconds, which is the difference between testing four ideas and forty.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 DUMP_MAXK=256
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
CKPT=$CY/trained_models/cyolo_sb/best_model.pt
OUT=/scratch/pmohseni/omr/cand_test; mkdir -p "$OUT"
for TIER in room do rp_synth; do
    [ -f "$OUT/$TIER.npz" ] && { echo "##### $TIER present, skipping"; continue; }
    export DUMP_OUT="$OUT/$TIER.npz"
    echo ""; echo "##### $TIER"
    python extensions/hooks/run_eval_dump.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/${TIER}_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |\[DUMP\]|rror"
done
echo ""; echo "Job finished at $(date)"
