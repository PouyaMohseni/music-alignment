#!/bin/bash
#SBATCH --job-name=v13-f1-combined
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v13_f1_combined-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v13_f1_combined-%j.log

# F1: v13 architecture + repeat-aware GT + MIDI->audio distillation (E2/E3)
# + soft-DTW position-trajectory consistency (C2's mechanism). All train-time
# only -- eval.py is v13's own, unmodified, no MIDI at inference.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

OUT=/scratch/pmohseni/results/v13_f1_combined
mkdir -p $OUT

RESUME_FLAG=""
if ls $OUT/checkpoint_epoch*.pt 2>/dev/null | grep -q .; then
    LATEST=$(ls $OUT/checkpoint_epoch*.pt | sort | tail -1)
    echo "Resuming from $LATEST"
    RESUME_FLAG="--resume $LATEST"
fi

python -m mymodel.v13_f1_combined.train \
    --config configs/v13_f1_combined.yaml \
    train.out_dir=$OUT \
    $RESUME_FLAG

echo ""
echo "Training done. Running test eval (v13's own eval.py, unmodified -- no MIDI at inference)..."
python -m mymodel.v13_mert_unet.eval \
    --checkpoint $OUT/best_model.pt \
    --config     configs/v13_f1_combined.yaml \
    --split      test \
    --out_dir    $OUT/eval

echo "Job finished at $(date)"
