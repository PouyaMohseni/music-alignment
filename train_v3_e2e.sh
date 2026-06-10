#!/bin/bash
#SBATCH --job-name=music-align-v3e2e
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_e2e/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_e2e/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v3_e2e

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

# Variant C: end-to-end fine-tuning.
# Encoder LoRA adapters (lr=5e-6) warm-started from v1_nce2 (30K NCE steps).
# Head (lr=1e-4) warm-started from v3_all/checkpoint_002000 (best full-seq head).
# Uses all-performances processed dataset for maximum data.
python -m mymodel.v3_e2e.train \
  --config configs/v3_e2e.yaml \
  data.processed_root=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/processed_all

echo "Job finished at $(date)"
