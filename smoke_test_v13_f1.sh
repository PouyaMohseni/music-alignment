#!/bin/bash
#SBATCH --job-name=smoke-v13-f1
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_v13_f1-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_v13_f1-%j.log

set -euo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
python3 scripts/smoke_test_v13_f1.py
