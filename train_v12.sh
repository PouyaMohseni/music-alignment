#!/bin/bash
#SBATCH --job-name=music-v12
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v12_mert_align-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v12_mert_align-%j.log

# v12: MERT-v1-95M audio + ResNet18 score columns + InfoNCE + expected-position loss
# Only the AlignmentHead (~400K params) is trained; encoders stay frozen.
# Target: >85% pct@0.5s on MSMD test split.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1   # MERT is already cached; no download needed

OUT=results/v12_mert_align
mkdir -p $OUT

RESUME_FLAG=""
if [ -f "$OUT/latest.pt" ]; then
    echo "Resuming from $OUT/latest.pt"
    RESUME_FLAG="--resume $OUT/latest.pt"
fi

python -m mymodel.v12_mert_align.train \
    --data_root data/MSMD/processed \
    --out       $OUT \
    --epochs    30 \
    --lr        3e-4 \
    --tau       0.07 \
    --w_infonce 1.0 \
    --w_expected 0.5 \
    --embed_dim 256 \
    --patience  10 \
    --device    cuda \
    $RESUME_FLAG

echo ""
echo "Training done. Running test eval..."
python -m mymodel.v12_mert_align.eval \
    --checkpoint $OUT/best_model.pt \
    --split test \
    --data_root data/MSMD/processed \
    --out $OUT/test_results.json

echo "Job finished at $(date)"
