#!/bin/bash
#SBATCH --job-name=music-align-v1-lora
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v1_lora/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v1_lora/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v1_lora

source .venv/bin/activate
pip install -q peft

export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"
echo "peft:  $(python -c 'import peft; print(peft.__version__)')"

python -m mymodel.v1_baseline.train \
  --config configs/v1_lora.yaml \
  train.steps=20000 \
  train.batch_size=4 \
  data.num_workers=4

echo "Job finished at $(date)"
