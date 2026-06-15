#!/bin/bash
#SBATCH --job-name=v3e2e-long
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_e2e/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_e2e/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v3_e2e

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

# Longer end-to-end run. Resume from the 500-step checkpoint (loads both the
# encoder LoRA and head via strict=False), then train far longer.
# ~23s/step -> 24h fits roughly ~3500 more steps. Checkpoints every 500 so a
# time-limit kill still leaves usable checkpoints to evaluate.
python -m mymodel.v3_e2e.train \
  --config configs/v3_e2e.yaml \
  data.processed_root=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/processed_all \
  data.num_workers=0 \
  train.init_encoder_checkpoint=results/v3_e2e/checkpoint_000500.pt \
  train.steps=4000 \
  optim.warmup_steps=100 \
  train.ckpt_every=500 \
  train.eval_every=500 \
  train.out_dir=results/v3_e2e_long

echo "Job finished at $(date)"
