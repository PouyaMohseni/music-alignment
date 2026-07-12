#!/bin/bash
#SBATCH --job-name=smoke-f2-ensemble
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f2_ensemble-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f2_ensemble-%j.log

set -euo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
python -m mymodel.f2_ensemble.eval --split test --limit 3
