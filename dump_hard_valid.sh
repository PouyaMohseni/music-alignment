#!/bin/bash
#SBATCH --job-name=hardval
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/hardval-%j.log
# A validation set as hard as the target.
#
# Reverberant synthetic validation has a detector argmax of 90.7 where room has
# 80.0, so it is a much easier problem -- and a higher-capacity selector can win
# there by fitting slack real audio does not have. Over seven variants,
# validation rank is now ANTI-correlated with room rank (Spearman -0.39): it
# picks vel_feat, worth 90.0, over feat_wide, worth 94.0.
#
# Only the VALIDATION set needs to get harder; the training dumps stay as they
# are. Three SNRs, 19 pieces each, and we keep whichever lands nearest 80.
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
export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey IR_SEED=7 IR_PROB=1.0
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
O=/scratch/pmohseni/omr/candhard; mkdir -p "$O"
for SNR in 20 12 6; do
    [ -f "$O/valid_snr$SNR.npz" ] && { echo "##### snr$SNR present"; continue; }
    export IR_SNR=$SNR DUMP_OUT="$O/valid_snr$SNR.npz"
    echo ""; echo "##### validation, room + noise at ${SNR} dB SNR"
    python extensions/hooks/run_eval_dump.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_valid" --split_files "$DATA/split_files/valid_c0_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |\[DUMP\]|\[IR\]|rror"
done
echo ""; echo "Job finished at $(date)"
