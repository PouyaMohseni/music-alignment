#!/bin/bash
#SBATCH --job-name=music-v10
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v10_mert_unet-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v10_mert_unet-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v10_mert_unet

source .venv/bin/activate

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
EMB=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/mert_emb

echo "=== Step 1: Precompute MERT embeddings ==="
python -m mymodel.v10_mert_unet.precompute \
    --processed $PROC \
    --out       $EMB \
    --fps       20

echo "=== Step 2: Train v10 ==="
python -m mymodel.v10_mert_unet.train \
    --config configs/v10_mert_unet.yaml \
    data.processed_root=$PROC \
    data.mert_emb_root=$EMB

echo "Training finished at $(date). Running eval..."

CKPT=$(ls results/v10_mert_unet/checkpoint_*.pt 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found after training"; exit 1
fi
echo "Evaluating: $CKPT"

echo "=== Step 3: Eval v10 ==="
python -m mymodel.v10_mert_unet.eval \
    --checkpoint $CKPT \
    --config     configs/v10_mert_unet.yaml \
    --split      test \
    --processed  $PROC \
    --mert_emb   $EMB \
    --out_dir    results/v10_mert_unet/eval

echo "Job finished at $(date)"
