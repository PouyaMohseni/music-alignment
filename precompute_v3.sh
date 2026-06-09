#!/bin/bash
#SBATCH --job-name=v3-precompute
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_fullseq/precompute-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_fullseq/precompute-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v3_fullseq

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Cache MERT + ViT embeddings using the LoRA-ADAPTED encoders from v1_nce2
# (not raw frozen), so the cached features reflect the domain adaptation.
python -m mymodel.v3_fullseq.precompute \
  --processed data/MSMD/processed \
  --config    configs/v3_fullseq.yaml \
  --init_checkpoint results/v1_nce2/checkpoint_020000.pt \
  --lora_rank 4 \
  --out       /lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_lora

echo "Job finished at $(date)"
