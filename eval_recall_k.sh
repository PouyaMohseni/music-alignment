#!/bin/bash
#SBATCH --job-name=recallk
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/recallk-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import cv2, scipy, numpy, torch" || { echo "FATAL: env broken"; exit 1; }
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-} PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python extensions/analysis/recall_at_k.py --dump /scratch/pmohseni/omr/candf256/room.npz
