#!/bin/bash
#SBATCH --job-name=music-align-v2
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=22:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v2_crossattn/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v2_crossattn/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v2_crossattn

source .venv/bin/activate

export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

# Warm-start from the best NCE checkpoint so cross-attention layers
# start with good encoder embeddings rather than random
python -m mymodel.v2_crossattn.train \
  --config configs/v2_crossattn.yaml \
  train.init_checkpoint=results/v1_nce2/checkpoint_020000.pt

echo "Job finished at $(date)"
