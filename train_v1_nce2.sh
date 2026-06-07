#!/bin/bash
#SBATCH --job-name=music-align-nce2
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v1_nce2/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v1_nce2/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v1_nce2

source .venv/bin/activate

export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

# Continue NCE from where we left off (checkpoint_010000 = 10K NCE steps done)
# Run 20K more steps with same config — NCE is still learning, not yet plateaued
python -m mymodel.v1_baseline.train \
  --config configs/v1_lora.yaml \
  train.steps=20000 \
  train.batch_size=4 \
  train.grad_accum_steps=4 \
  train.init_checkpoint=results/v1_nce/checkpoint_010000.pt \
  train.out_dir=results/v1_nce2 \
  data.num_workers=4 \
  data.manifest_path=data/MSMD/processed/manifest.jsonl \
  loss.nce_only=true \
  optim.lr=5.0e-5 \
  optim.warmup_steps=0

echo "Job finished at $(date)"
