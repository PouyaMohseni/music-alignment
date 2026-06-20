#!/bin/bash
# Submit all v5 training variants at once.
# Usage: bash submit_v5_all.sh

cd /project/def-ichiro/pmohseni/music-alignment

mkdir -p results/v5b_large results/v5c_noxattn results/v5d_long

echo "Submitting v5b (large LSTM)..."
sbatch train_v5b.sh

echo "Submitting v5c (no cross-attention)..."
sbatch train_v5c.sh

echo "Submitting v5d (long training)..."
sbatch train_v5d.sh

echo ""
echo "All submitted. Check status with: squeue -u $USER"
echo ""
echo "When done tomorrow, run:  bash eval_v5_all.sh"
