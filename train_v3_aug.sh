#!/bin/bash
#SBATCH --job-name=v3-aug-train
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_aug-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_aug-%j.log

# Retrain v3_fullseq on multi-tempo augmented data.
# Requires precompute_aug.sh to have completed (data/MSMD/embeddings_aug/).
# Architecture: frozen MERT + ViT, cross-attention head only.
# Training objective: expected_distance_loss (distance-aware localization).
# Data: 467 pieces × 11 tempos = ~3900 train samples.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export TRANSFORMERS_CACHE=/scratch/pmohseni/hf_cache
export HF_HOME=/scratch/pmohseni/hf_cache

mkdir -p results/v3_aug

python -m mymodel.v3_fullseq.train \
    --config configs/v3_aug.yaml

echo "Job finished at $(date)"
