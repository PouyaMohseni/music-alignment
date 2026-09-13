#!/bin/bash
#SBATCH --job-name=dumpenc
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-6
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dumpenc_%x_%a-%A.log
# Candidate dumps from a retrained encoder arm, so its decision model can be
# trained on its own proposals and cross-validated like every other row.
#
#   sbatch dump_encoders.sh lstm|mamba|cnnmamba|dinov2|cnn
#
#   task 0 room (FEATK 256, clean)   1 valid   2-6 train shards c0-c4 (real IR)
set -uo pipefail
ARM=${1:?usage: dump_encoders.sh lstm|mamba|cnnmamba|dinov2|cnn}
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 DUMP_MAXK=256
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK

case $ARM in
  lstm)     ROOT=/scratch/pmohseni/mambaenc/cyolo_sb_lstm ;;
  mamba)    ROOT=/scratch/pmohseni/mambaenc/cyolo_sb_mamba;    export MAMBA_ENC=1 MAMBA_MODE=tower ;;
  cnnmamba) ROOT=/scratch/pmohseni/mambaenc/cyolo_sb_cnnmamba; export MAMBA_ENC=1 MAMBA_MODE=seq ;;
  dinov2)   ROOT=/scratch/pmohseni/dinoenc/cyolo_sb_dinov2;    export DINO_ENC=1 ;;
  cnn)      ROOT=/scratch/pmohseni/dinoenc/cyolo_sb_cnn ;;
  *) echo "FATAL: unknown arm $ARM"; exit 1 ;;
esac
CKPT=$(find "$ROOT/params" -name best_model.pt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "FATAL: no best_model.pt under $ROOT/params"; exit 1; }
echo "##### arm=$ARM ckpt=$CKPT"

J=${SLURM_ARRAY_TASK_ID}
case $J in
  0) OUT=room;  DIR=msmd_rp;    SPLIT=room_split;     FK=256; unset IR_PATH ;;
  1) OUT=valid; DIR=msmd_valid; SPLIT=valid_c0_split; FK=128 ;;
  *) C=$((J - 2)); OUT=train_c$C; DIR=msmd_train; SPLIT=train_c${C}_split; FK=128 ;;
esac
[ "$J" -ne 0 ] && export IR_PATH=/scratch/pmohseni/ir_bank IR_SEED=0 IR_PROB=1.0
O=/scratch/pmohseni/omr/cand_enc_$ARM; mkdir -p "$O"
[ -f "$O/$OUT.npz" ] && { echo "present: $O/$OUT.npz"; exit 0; }
export DUMP_FEATK=$FK DUMP_OUT="$O/$OUT.npz"
echo "##### $ARM $OUT ir=${IR_PATH:-none} $(date)"
python extensions/hooks/run_eval_dump.py --param_path "$CKPT" \
    --test_dirs "$DATA/$DIR" --split_files "$DATA/split_files/$SPLIT.yaml" \
    --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
    | grep -E "^<= |\[DUMP\]|\[FEAT\]|\[MAMBA\]|\[DINO\]|rror|Traceback"
ls -l "$O/$OUT.npz"
