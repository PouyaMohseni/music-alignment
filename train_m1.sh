#!/bin/bash
#SBATCH --job-name=m1-monotonic
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/m1_monotonic-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/m1_monotonic-%j.log

# M1: monotonic cross-modal alignment (see M1.md). Reuses D1's MERT-audio +
# CNN/transformer-score towers (cpjku_fmt strips + precomputed whole-piece
# MERT), but supervised with the forward-sum monotonic-alignment objective +
# annealed beta-binomial prior over ONSET columns, decoded by monotonic Viterbi.
# Same main .venv / precomputed-MERT setup as train_d1_align_matrix.sh (no
# madmom needed). Phase-1 overfit-one-piece proof must have passed first.

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/m1_monotonic
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

RESUME_FLAG=""
if [ -f results/m1_monotonic/checkpoint_latest.pt ]; then
    echo "Found existing checkpoint -- resuming."
    RESUME_FLAG="--resume"
fi

python -m mymodel.m1_monotonic.train \
    --config configs/m1_monotonic.yaml \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval (monotonic Viterbi, repeat-stratified)..."
python -m mymodel.m1_monotonic.eval \
    --config configs/m1_monotonic.yaml \
    --checkpoint results/m1_monotonic/best_model.pt \
    --split test --repeat_stratified

echo "Job finished at $(date)"
