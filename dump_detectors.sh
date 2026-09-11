#!/bin/bash
#SBATCH --job-name=dumpdet
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-15
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dumpdet_%a-%A.log
# Candidate dumps from the two OTHER released detectors, so the decoder can be
# tested on each: CYOLO (notes only, no bar/system heads) and CYOLO-SB+A
# (trained with extra proprietary recordings). Exactly dump_features.sh's
# recipe -- real-IR audio for train/valid, room clean, FEATK 128/256 -- with
# only --param_path changed.
#
#   task = 8 * detector + j;  j: 0 room  1 valid  2-7 train c0-c5
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 DUMP_MAXK=256
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
DETS=(cyolo cyolo_sb_a)
I=${SLURM_ARRAY_TASK_ID}
D=${DETS[$((I / 8))]}
J=$((I % 8))
O=/scratch/pmohseni/omr/cand_$D; mkdir -p "$O"
case $J in
  0) OUT=room;  DIR=msmd_rp;    SPLIT=room_split;     FK=256; unset IR_PATH ;;
  1) OUT=valid; DIR=msmd_valid; SPLIT=valid_c0_split; FK=128 ;;
  *) C=$((J - 2)); OUT=train_c$C; DIR=msmd_train; SPLIT=train_c${C}_split; FK=128 ;;
esac
[ "$J" -ne 0 ] && export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey IR_SEED=0 IR_PROB=1.0
[ -f "$O/$OUT.npz" ] && { echo "present: $O/$OUT.npz"; exit 0; }
export DUMP_FEATK=$FK DUMP_OUT="$O/$OUT.npz"
echo "##### $D  $OUT  ir=${IR_PATH:-none}  $(date)"
python extensions/hooks/run_eval_dump.py \
    --param_path "$CY/trained_models/$D/best_model.pt" \
    --test_dirs "$DATA/$DIR" --split_files "$DATA/split_files/$SPLIT.yaml" \
    --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
    | grep -E "^<= |\[DUMP\]|\[FEAT\]|\[IR\]|rror|Traceback"
ls -l "$O/$OUT.npz"
