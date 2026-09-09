#!/bin/bash
#SBATCH --job-name=cmte
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cmte-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "########## room ##########"
python extensions/analysis/committee.py --dump /scratch/pmohseni/omr/candf256/room.npz --procs 8
echo ""; echo "########## held-out snr12 (80 pieces, real power) ##########"
python extensions/analysis/committee.py --dump /scratch/pmohseni/omr/candhv256/valid_snr12.npz --procs 8
