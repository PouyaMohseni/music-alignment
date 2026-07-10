#!/bin/bash
#SBATCH --job-name=smoke-d3
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_d3-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_d3-%j.log

set -euo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
python3 scripts/smoke_test_d3.py
