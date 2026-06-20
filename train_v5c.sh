#!/bin/bash
#SBATCH --job-name=music-v5c
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v5c_noxattn/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v5c_noxattn/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v5c_noxattn

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# No warm-start for no-cross-attn variant (v3 cross-attn weights would be wasted)
# v5c: LSTM only, no cross-attention — simpler, tests if xattn is actually needed
python -m mymodel.v5_recurrent.train \
  --config configs/v5c_noxattn.yaml \
  data.emb_root=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_all_tar \
  data.processed_root=data/MSMD/processed_all

echo "Job finished at $(date)"
