#!/bin/bash
#SBATCH --job-name=v3-all
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_all/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_all/slurm-%j.log

# Stage 3: train v3 on ALL performances (~13x data) from tar-shard embeddings.
# More data delays the overfitting that capped the single-performance run at
# step ~1500, so we run longer and checkpoint frequently.
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v3_all

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SCRATCH=/lustre07/scratch/pmohseni/music-alignment/data/MSMD

python -m mymodel.v3_fullseq.train \
  --config configs/v3_fullseq.yaml \
  data.emb_root=$SCRATCH/embeddings_all_tar \
  data.processed_root=$SCRATCH/processed_all \
  train.steps=20000 \
  train.ckpt_every=1000 \
  train.eval_every=500 \
  train.out_dir=results/v3_all

echo "Job finished at $(date)"
