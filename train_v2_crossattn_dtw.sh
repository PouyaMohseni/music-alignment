#!/bin/bash
#SBATCH --job-name=v2-crossattn-dtw
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=22:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v2_crossattn_dtw-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v2_crossattn_dtw-%j.log

# v2 DTW phase (real cross-attention + DTW alignment training). The
# original train_v2.sh only ever ran the NCE-only warmup (configs/
# v2_crossattn.yaml has nce_only:true and out_dir:results/v2_nce) -- the
# "separate job" its own comments describe as following the warmup never
# existed. This script runs that job for the first time, using
# configs/v2_crossattn_dtw.yaml (nce_only:false, warm-started from the
# completed warmup checkpoint results/v2_nce/checkpoint_010000.pt).

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v2_crossattn_dtw

source .venv/bin/activate

export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

# If this job was resubmitted after a timeout, resume from its OWN latest
# checkpoint instead of re-starting from the NCE warmup checkpoint.
OVERRIDE=""
LATEST_CKPT=$(find results/v2_crossattn_dtw -maxdepth 1 -name "checkpoint_*.pt" -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming DTW phase from its own checkpoint: $LATEST_CKPT"
    OVERRIDE="train.init_checkpoint=$LATEST_CKPT"
else
    echo "Starting DTW phase from the NCE warmup checkpoint (results/v2_nce/checkpoint_010000.pt)"
fi

python -m mymodel.v2_crossattn.train \
  --config configs/v2_crossattn_dtw.yaml \
  $OVERRIDE

echo "Job finished at $(date)"
