#!/bin/bash
#SBATCH --job-name=ambig
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/ambig-%j.log
# selected_room (the 91.4 model) first: it must reproduce the page's 355 / 18 /
# 59.2% / 0.000 before the nbrp64_s0 line means anything.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
T=/scratch/pmohseni/omr/traj
python extensions/analysis/ambiguity.py "$T/selected_room.traj.npz" \
    "$T/velp8_room.traj.npz" "$T/nbrp64s0_room.traj.npz"
