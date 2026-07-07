#!/bin/bash
#SBATCH --job-name=c1-visual-grounding
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/c1_visual_grounding-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/c1_visual_grounding-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/c1_visual_grounding

source .venv/bin/activate

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed

echo "=== Train C1: audio-visual grounding via cross-attention ==="
# Auto-resume if a checkpoint already exists (same pattern as train_v11.sh)
RESUME_FLAG=""
if ls results/c1_visual_grounding/checkpoint_epoch*.pt 1>/dev/null 2>&1; then
    echo "Found existing checkpoint — resuming."
    RESUME_FLAG="--resume"
fi
python -m mymodel.c1_visual_grounding.train \
    --config configs/c1_visual_grounding.yaml \
    data.processed_root=$PROC \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval..."

CKPT=results/c1_visual_grounding/best_model.pt
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls results/c1_visual_grounding/checkpoint_epoch*.pt 2>/dev/null | sort | tail -1)
fi
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found after training"; exit 1
fi
echo "Evaluating: $CKPT"

echo "=== Eval C1 on test split ==="
python -m mymodel.c1_visual_grounding.eval \
    --checkpoint $CKPT \
    --config     configs/c1_visual_grounding.yaml \
    --split      test \
    --processed  $PROC \
    --out_dir    results/c1_visual_grounding/eval

echo "Job finished at $(date)"
