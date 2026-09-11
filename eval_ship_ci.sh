#!/bin/bash
#SBATCH --job-name=shipci
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/shipci-%j.log
# Piece-clustered CIs on room for the ladder with nbrp64_s0 on top, on the
# FEATK=256 dump the harness supplies (the dump that reproduces 94.89).
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python extensions/analysis/ladder_ci.py --dump /scratch/pmohseni/omr/candf256/room.npz \
    --ship /scratch/pmohseni/omr/scorer/nbr/nbrp64_s0.pt
