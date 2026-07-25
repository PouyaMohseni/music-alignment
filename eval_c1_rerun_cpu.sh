#!/bin/bash
#SBATCH --job-name=eval-c1-rerun-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_c1_rerun-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_c1_rerun-%j.log

# Re-evaluate c1_visual_grounding's EXISTING best_model.pt after fixing
# eval.py's decode (removed the sigmoid-style >=0.5 threshold + strip-
# midpoint fallback, which was firing on ~83% of frames because c1's
# heatmap is a softmax over score patches, not an independent-per-pixel
# sigmoid map -- see the cross-attention bug audit). No retraining needed,
# training itself converged fine.

echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed

echo "=== Re-eval c1_visual_grounding (eval.py decode fix) ==="
python -m mymodel.c1_visual_grounding.eval \
    --checkpoint results/c1_visual_grounding/best_model.pt \
    --config     configs/c1_visual_grounding.yaml \
    --split      test \
    --processed  $PROC \
    --out_dir    results/c1_visual_grounding/eval

echo "Job finished at $(date)"
