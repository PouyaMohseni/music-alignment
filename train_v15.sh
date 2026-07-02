#!/bin/bash
#SBATCH --job-name=music-v15
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v15_mert_mlp-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v15_mert_mlp-%j.log

# v15: MERT pre-computed features → MLP(768→256→32) → ConditionalUNet + Dice + BPTT

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

OUT=/scratch/pmohseni/results/v15_mert_mlp
mkdir -p $OUT

RESUME_FLAG=""
if ls $OUT/checkpoint_epoch*.pt 2>/dev/null | grep -q .; then
    LATEST=$(ls $OUT/checkpoint_epoch*.pt | sort | tail -1)
    echo "Resuming from $LATEST"
    RESUME_FLAG="--resume $LATEST"
fi

python -m mymodel.v13_mert_unet.train \
    --config configs/v15_mert_mlp.yaml \
    train.out_dir=$OUT \
    $RESUME_FLAG

echo ""
echo "Training done. Running test eval..."
python -m mymodel.v13_mert_unet.eval \
    --checkpoint $OUT/best_model.pt \
    --config     configs/v15_mert_mlp.yaml \
    --split      test \
    --out_dir    $OUT/eval

echo "Job finished at $(date)"
