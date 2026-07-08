#!/bin/bash
#SBATCH --job-name=d1-align-matrix
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/d1_align_matrix-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/d1_align_matrix-%j.log

# D1: dense frame x column alignment with global monotonic DTW decode (see D1.md).
# Two-tower MERT-audio + CNN-score -> similarity matrix, dense per-frame CE +
# downsampled soft-DTW, decoded by DTW. Trains on cpjku_fmt strips + precomputed
# whole-piece MERT (prereq: precompute_mert_trainval.sh must have finished).

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/d1_align_matrix
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

RESUME_FLAG=""
if [ -f results/d1_align_matrix/checkpoint_latest.pt ]; then
    echo "Found existing checkpoint -- resuming."
    RESUME_FLAG="--resume"
fi

python -m mymodel.d1_align_matrix.train \
    --config configs/d1_align_matrix.yaml \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval (offline DTW)..."
python -m mymodel.d1_align_matrix.eval \
    --config configs/d1_align_matrix.yaml \
    --checkpoint results/d1_align_matrix/best_model.pt \
    --split test

echo ""
echo "=== also evaluating causal OLTW decode ==="
python -m mymodel.d1_align_matrix.eval \
    --config configs/d1_align_matrix.yaml \
    --checkpoint results/d1_align_matrix/best_model.pt \
    --split test --online

echo "Job finished at $(date)"
