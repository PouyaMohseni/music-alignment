#!/bin/bash
#SBATCH --job-name=ceiling
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/ceiling-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "########## room ##########"
python extensions/analysis/ceiling_anatomy.py --dump /scratch/pmohseni/omr/candf/room.npz
echo ""; echo "########## held-out snr12 ##########"
python extensions/analysis/ceiling_anatomy.py --dump /scratch/pmohseni/omr/candhv/valid_snr12.npz
