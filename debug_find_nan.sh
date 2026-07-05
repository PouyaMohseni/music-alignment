#!/bin/bash
#SBATCH --job-name=debug-find-nan
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/debug_find_nan-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/debug_find_nan-%j.log

set -euo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
python scripts/debug_find_nan_piece.py
