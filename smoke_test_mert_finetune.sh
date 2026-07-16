#!/bin/bash
#SBATCH --job-name=smoke-mert-finetune
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_mert_finetune-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_mert_finetune-%j.log

# Verifies the new v11-mert-finetune pipeline (live, fine-tunable MERT-v1-95M
# inside the CB_TA-faithful FiLM+UNet+LSTM architecture) actually works
# end-to-end -- one real forward+backward BPTT chunk, checking gradients
# genuinely reach MERT's unfrozen parameters -- BEFORE committing a full,
# expensive (24h+) training job to it. MERT model construction stalled badly
# on the shared login node's CPU (>10 min, never completed); running this on
# an actual GPU compute node instead.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/smoke_test_mert_finetune.py

echo "Job finished at $(date)"
