#!/bin/bash
#SBATCH --job-name=hv-noisy
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/hv_noisy-%j.log
# Make validation as HARD as room, measured by the detector's own argmax.
#
#   ROOM (the target)        80.0%
#   old valid, room 19pc     90.7%
#   held-out 80pc, room      97.3%   <- the new split is FURTHER away
#
# The 80 pieces come out of msmd_train, which cyolo_sb was trained on, so the
# detector is near-perfect on that music whatever acoustics we add. The split
# fixes the sample-size problem (40,594 frames vs 3,728) and cannot fix the
# difficulty problem; noise is the only lever left.
#
# Previous attempt died on rng.normal() -- that is numpy's API and `rng` is a
# stdlib random.Random. Now a numpy generator seeded from the same per-piece
# key, so the noise stays deterministic per piece like the room choice.
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
export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey IR_PROB=1.0 IR_SEED=7
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
O=/scratch/pmohseni/omr/candhv; mkdir -p "$O"
for SNR in 12 6 3 0.5; do
    T="$O/valid_snr$SNR.npz"
    [ -s "$T" ] && { echo "##### snr$SNR present"; continue; }
    export IR_SNR=$SNR DUMP_OUT="$T"
    echo ""; echo "##### held-out 80 pieces, room + noise at ${SNR} dB"
    python extensions/hooks/run_eval_dump.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_train" --split_files "$DATA/split_files/hv_valid_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |\[DUMP\]|\[IR\]|rror|Traceback"
done
echo ""; echo "Job finished at $(date)"
