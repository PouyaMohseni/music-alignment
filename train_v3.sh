#!/bin/bash
#SBATCH --job-name=music-align-v3
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_fullseq/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_fullseq/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v3_fullseq

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

python -m mymodel.v3_fullseq.train \
  --config configs/v3_fullseq.yaml \
  data.emb_root=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_lora

echo "Job finished at $(date)"
