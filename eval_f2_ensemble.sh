#!/bin/bash
#SBATCH --job-name=eval-f2-ensemble
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_f2_ensemble-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_f2_ensemble-%j.log

# F2: full 94-piece eval of the zero-retrain v13+v14+v15 heatmap ensemble.
# 3-piece smoke test showed 72.9% pct@0.5s vs. 66.1/66.0/66.4% individually.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

python -m mymodel.f2_ensemble.eval --split test

echo "Job finished at $(date)"
