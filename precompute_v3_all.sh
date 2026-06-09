#!/bin/bash
#SBATCH --job-name=v3-precompute-all
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_fullseq/precompute-all-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_fullseq/precompute-all-%j.log

# Stage 2: cache LoRA-adapted embeddings for ALL performances into TAR SHARDS
# (~6000 pieces -> ~12 shards, not 6000 files -> no inode blowup). Tile
# embeddings are cached per shared strip so ViT runs once per piece, not per perf.
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SCRATCH=/lustre07/scratch/pmohseni/music-alignment/data/MSMD

python -m mymodel.v3_fullseq.precompute \
  --processed       $SCRATCH/processed_all \
  --config          configs/v3_fullseq.yaml \
  --init_checkpoint results/v1_nce2/checkpoint_020000.pt \
  --lora_rank       4 \
  --shard_size      500 \
  --out             $SCRATCH/embeddings_all_tar

echo "Job finished at $(date)"
