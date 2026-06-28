#!/bin/bash
#SBATCH --job-name=music-v11
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v11_cpjku_fullstrip-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v11_cpjku_fullstrip-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v11_cpjku_fullstrip

source .venv/bin/activate

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed

echo "=== Train v11: CPJKU full-strip BPTT ==="
# Auto-resume if a checkpoint already exists (supports multi-job continuation)
RESUME_FLAG=""
if ls results/v11_cpjku_fullstrip/checkpoint_epoch*.pt 1>/dev/null 2>&1; then
    echo "Found existing checkpoint — resuming."
    RESUME_FLAG="--resume"
fi
python -m mymodel.v11_cpjku_fullstrip.train \
    --config configs/v11_cpjku_fullstrip.yaml \
    data.processed_root=$PROC \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval..."

CKPT=results/v11_cpjku_fullstrip/best_model.pt
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls results/v11_cpjku_fullstrip/checkpoint_epoch*.pt 2>/dev/null | sort | tail -1)
fi
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found after training"; exit 1
fi
echo "Evaluating: $CKPT"

echo "=== Eval v11 on test split ==="
python -m mymodel.v11_cpjku_fullstrip.eval \
    --checkpoint $CKPT \
    --config     configs/v11_cpjku_fullstrip.yaml \
    --split      test \
    --processed  $PROC \
    --out_dir    results/v11_cpjku_fullstrip/eval

echo "Job finished at $(date)"
