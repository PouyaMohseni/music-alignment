#!/bin/bash
#SBATCH --job-name=eval-d1-band
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_d1_band-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_d1_band-%j.log

# Re-evaluate D1's ALREADY-TRAINED checkpoint (job 64908703) with the newly
# banded DTW decode vs. the original unbanded decode, to measure whether
# Sakoe-Chiba banding actually reduces the bimodal-error pattern seen in the
# first run (median 3.85s, mean 9.7s -- close when right, catastrophic when
# wrong). No retraining -- this is decode-only, fast.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

CKPT=results/d1_align_matrix/best_model.pt

echo "=== banded (band_frac=0.15) ==="
python -m mymodel.d1_align_matrix.eval \
    --config configs/d1_align_matrix.yaml --checkpoint "$CKPT" --split test --band_frac 0.15

echo ""
echo "=== unbanded (original behavior) ==="
python -m mymodel.d1_align_matrix.eval \
    --config configs/d1_align_matrix.yaml --checkpoint "$CKPT" --split test --band_frac -1

echo ""
echo "=== tighter band (band_frac=0.05) ==="
python -m mymodel.d1_align_matrix.eval \
    --config configs/d1_align_matrix.yaml --checkpoint "$CKPT" --split test --band_frac 0.05

echo "Job finished at $(date)"
