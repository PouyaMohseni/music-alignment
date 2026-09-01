#!/bin/bash
#SBATCH --job-name=hv-dump
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/hv_dump-%j.log
# A validation split that is actually a split.
#
# msmd_valid is 19 pieces, and every validation set we have is those same 19
# under a different acoustic condition -- so a one-point difference is a handful
# of frames on a handful of recordings, which is why the ranking flipped sign
# once variants differed in capacity rather than training data (Spearman -0.39
# against room over seven variants).
#
# 80 pieces held out of the 353, stratified by composer at the PIECE level, so
# the mix still resembles the Bach-heavy target (39.6% vs 40.0%). Costs 23% of
# the training data, which is cheap against a 10-22k parameter selector.
#
# Validation also gets NOISE on top of the room, at three SNRs, because
# reverberant synthetic alone has an argmax of 90.7 where room has 80.0.
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
export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey IR_PROB=1.0
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
CKPT=$CY/trained_models/cyolo_sb/best_model.pt
O=/scratch/pmohseni/omr/candhv; mkdir -p "$O"

run () { local out=$1 split=$2
    [ -f "$out" ] && { echo "##### $(basename $out) present"; return; }
    export DUMP_OUT="$out"
    echo ""; echo "##### $(basename $out)  seed=$IR_SEED snr=${IR_SNR}"
    python extensions/hooks/run_eval_dump.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_train" --split_files "$DATA/split_files/$split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |\[DUMP\]|\[IR\]|\[FEAT\]|rror"; }

# held-out validation at three difficulties; the "<= 0.5" line printed by each
# is the DETECTOR's own score there, so we can pick the SNR whose difficulty
# matches room (80.0) rather than guessing
export IR_SEED=7
for SNR in 0 12 6; do
    export IR_SNR=$SNR; run "$O/valid_snr$SNR.npz" hv_valid_split
done
# training half, same room recipe as before
export IR_SEED=0 IR_SNR=0
for i in 0 1 2 3 4; do run "$O/train_c$i.npz" hv_train_c${i}_split; done
echo ""; echo "Job finished at $(date)"; du -sh "$O"
