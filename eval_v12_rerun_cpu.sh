#!/bin/bash
#SBATCH --job-name=eval-v12-rerun-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v12_rerun-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v12_rerun-%j.log

# Re-evaluate v12_mert_align's EXISTING best_model.pt after fixing the
# swapped backtrack-direction bug in mymodel/v12_mert_align/dtw.py -- no
# retraining needed, the model/checkpoint was fine, only the DTW decode was
# broken (see the cross-attention bug audit).

echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

OUT=results/v12_mert_align

echo "=== Re-eval v12_mert_align (dtw.py backtrack fix) ==="
python -m mymodel.v12_mert_align.eval \
    --checkpoint $OUT/best_model.pt \
    --split test \
    --data_root data/MSMD/processed \
    --out $OUT/test_results.json

echo "Job finished at $(date)"
