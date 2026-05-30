#!/bin/bash
#SBATCH --job-name=music-align-v1
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v1_baseline/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v1_baseline/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

# Go to project
cd /project/def-ichiro/pmohseni/music-alignment

# Activate venv
source .venv/bin/activate

# Offline mode (HF models pre-cached in project space)
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

# Train
python -m mymodel.v1_baseline.train \
  train.steps=20000 \
  train.batch_size=16 \
  data.num_workers=8 \
  train.eval_every=500 \
  train.ckpt_every=2000

echo "Job finished at $(date)"
