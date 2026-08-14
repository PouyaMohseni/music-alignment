#!/bin/bash
#SBATCH --job-name=s2-sub
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/pmohseni/omr/logs/s2-sub-%j.log
# S2 subset canary -- 50 pieces, short queue, no a100 constraint so it starts
# FIRST. Its job is to fail fast and cheaply: if the loss does not move or the
# val column error does not fall here, the full run gets adjusted rather than
# burning 24 h of A100 to learn the same thing.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

python extensions/models/train_s2_align.py \
  --out /scratch/pmohseni/omr/s2_align_sub \
  --epochs 25 --chunk 64 --batch 4 --lr 3e-4 \
  --workers 3 --strip_scale 2 \
  --max_pieces 50 --max_minutes 180
