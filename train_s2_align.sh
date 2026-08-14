#!/bin/bash
#SBATCH --job-name=s2-align
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/pmohseni/omr/logs/s2-align-%j.log
# S2 -- learned monotonic alignment, FULL training set (354 pieces).
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1

python extensions/models/train_s2_align.py \
  --out /scratch/pmohseni/omr/s2_align_full \
  --epochs 40 --chunk 64 --batch 4 --lr 3e-4 \
  --workers 5 --strip_scale 2
