#!/bin/bash
#SBATCH --job-name=eval-d1-band-tight
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_d1_band_tight-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_d1_band_tight-%j.log

# eval_d1_band_compare.sh showed a clean monotonic trend: tighter band is
# strictly better (unbanded 12.3%/9.73s -> 0.15: 13.7%/5.81s -> 0.05: 16.9%/3.98s
# on pct@0.5s/mean-error). Testing even tighter to find where it plateaus or
# starts hurting (too tight forbids genuine large tempo deviations).

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
CKPT=results/d1_align_matrix/best_model.pt

for bf in 0.02 0.01; do
    echo "=== band_frac=$bf ==="
    python -m mymodel.d1_align_matrix.eval \
        --config configs/d1_align_matrix.yaml --checkpoint "$CKPT" --split test --band_frac $bf
    echo ""
done

echo "Job finished at $(date)"
