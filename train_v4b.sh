#!/bin/bash
#SBATCH --job-name=music-align-v4b
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v4b_pitch/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v4b_pitch/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v4b_pitch

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Python: $(which python)"

# Find latest v3_all checkpoint to warm-start from
V3_CKPT=$(ls results/v3_all/checkpoint_*.pt 2>/dev/null | sort | tail -1)
if [ -z "$V3_CKPT" ]; then
    echo "ERROR: no v3_all checkpoint found in results/v3_all/ — aborting"
    exit 1
fi
echo "Warm-starting from: $V3_CKPT"

# v4b: warm-started from v3_all + gentle pitch fusion (alpha=0.2, pw=0.3)
# Fixes v4's collapse (alpha=1.0 + random pitch heads + aggressive loss = 11.6s)
python -m mymodel.v4_pitch.train \
  --config configs/v4b_pitch.yaml \
  init_v3_checkpoint=$V3_CKPT \
  data.emb_root=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_lora_all

echo "Job finished at $(date)"
