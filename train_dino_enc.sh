#!/bin/bash
#SBATCH --job-name=dinoenc
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dinoenc_%x-%j.log
# The image-side twin of the audio encoder ablation.
#
#   sbatch train_dino_enc.sh dinov2   DINOv2-base patches replace the visual stem
#   sbatch train_dino_enc.sh cnn      the matched control, stem unchanged
#
# Both drop the page shift: a precomputed 52x52 patch map rolls only in whole
# 8-px steps and the note anchors are 11 px wide, so a sub-patch roll would
# desynchronise features from labels by most of a notehead. Tempo augmentation
# stays on for both. The LSTM arm launched yesterday had shifts on and is
# therefore NOT the control for this pair.
set -uo pipefail
ARM=${1:?usage: train_dino_enc.sh dinov2|cnn}
case "$ARM" in
  dinov2) export DINO_ENC=1 ;;
  cnn)    export DINO_ENC=0 ;;
  *) echo "FATAL: arm must be dinov2 or cnn"; exit 1 ;;
esac
export DINO_NOSHIFT=1
echo "Job started on $(hostname) at $(date), arm=$ARM"
nvidia-smi | head -12
CFG=${2:-cyolo_sb}
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
OUT=/scratch/pmohseni/dinoenc/${CFG}_${ARM}
mkdir -p "$OUT/params" "$OUT/runs"
PARAM_FLAG=""
LAST=$(find "$OUT/params" -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$LAST" ] && { LAST=$(readlink -f "$LAST"); echo "Resuming from $LAST"; PARAM_FLAG="--param_path $LAST"; }
IR=/scratch/pmohseni/ir_bank
echo "=== $ARM arm, config=$CFG, page shift off, IR on ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_dino.py \
    --train_sets "$DATA/msmd_train" --val_sets "$DATA/msmd_valid" \
    --config "$CY/cyolo_score_following/models/configs/${CFG}.yaml" \
    --augment --ir_path "$IR" \
    --dump_root "$OUT/params" --log_root "$OUT/runs" \
    --tag ${CFG}_${ARM} --num_workers 2 $PARAM_FLAG
STATUS=$?
find "$OUT/params" -name "*.pt" -printf "  ckpt %p (%s bytes)\n" 2>/dev/null | head -3
echo "exit $STATUS at $(date)"
exit $STATUS
