#!/bin/bash
#SBATCH --job-name=music-v5d
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v5d_long/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v5d_long/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v5d_long

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

V3_CKPT=$(ls results/v3_all/checkpoint_*.pt 2>/dev/null | sort | tail -1)
if [ -z "$V3_CKPT" ]; then echo "ERROR: no v3_all checkpoint" && exit 1; fi
echo "Warm-starting from: $V3_CKPT"

# v5d: same architecture as v5 but 16000 steps — tests if LSTM needs more time
python -m mymodel.v5_recurrent.train \
  --config configs/v5d_long.yaml \
  init_v3_checkpoint=$V3_CKPT \
  data.emb_root=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_all_tar \
  data.processed_root=data/MSMD/processed_all

echo "Job finished at $(date)"
