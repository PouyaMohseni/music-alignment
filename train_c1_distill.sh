#!/bin/bash
#SBATCH --job-name=c1-distill
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/c1_distill-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/c1_distill-%j.log

# C1 -- posteriorgram distillation on top of the RELEASED cyolo_sb (79.9).
#
# We do NOT retrain the baseline. trained_models/cyolo_sb/best_model.pt IS the
# 79.9 model and we verified it in our own harness, so fine-tuning from it makes
# the delta attributable to this loss alone with zero reproduction variance.
#
#   usage: sbatch train_c1_distill.sh [weight]

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi | head -12

W=${1:-1.0}
TAG=C1_distill_w${W}

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import torch,madmom,librosa,cv2" || { echo "FATAL: venv_cyolo incomplete"; exit 1; }

CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
POST=/scratch/pmohseni/amt_post_cyolo
OUT=/scratch/pmohseni/c1_distill/$TAG
mkdir -p "$OUT/params" "$OUT/runs"
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# dist_utils.py:16 sets rank from SLURM_PROCID without world_size, then dies.
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
# OpenMP is not fork-safe and dataset.py:301 uses get_context("fork").Pool(8).
export OMP_NUM_THREADS=1

export C1_BANK_MAP="$DATA/msmd_train=$POST/msmd_train;$DATA/msmd_valid=$POST/msmd_valid"
export C1_WEIGHT=$W

# Resume our own run if present, else warm-start from the RELEASED 79.9 model.
BASE=$CY/trained_models/cyolo_sb/best_model.pt
LAST=$(find "$OUT/params" -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LAST" ]; then
    LAST=$(readlink -f "$LAST"); echo "Resuming from $LAST"; PARAM_FLAG="--param_path $LAST"
else
    [ -s "$BASE" ] || { echo "FATAL: released cyolo_sb checkpoint missing: $BASE"; exit 1; }
    echo "Warm-starting from the RELEASED cyolo_sb (79.9): $BASE"
    PARAM_FLAG="--param_path $BASE"
fi

echo "=== $TAG: posteriorgram distillation, weight=$W ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_c1_distill.py \
    --train_sets "$DATA/msmd_train" \
    --val_sets   "$DATA/msmd_valid" \
    --config "$CY/cyolo_score_following/models/configs/cyolo_sb.yaml" \
    --augment \
    --ir_path /scratch/pmohseni/ir_bank \
    --dump_root "$OUT/params" \
    --log_root  "$OUT/runs" \
    --tag ${TAG} \
    --num_workers 2 \
    --num_epochs 20 \
    $PARAM_FLAG

echo ""; echo "Job finished at $(date)"
