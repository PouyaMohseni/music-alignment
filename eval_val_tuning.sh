#!/bin/bash
#SBATCH --job-name=valtune
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/valtune-%j.log
# Decoder constants chosen on the MSMD VALIDATION split (19 pieces, real-IR
# audio) -- disjoint from the test pieces and the standard place to tune --
# then room read once, plus a room sensitivity sweep reported as analysis only.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python extensions/analysis/heldout_tuning.py --tune /scratch/pmohseni/omr/candf/valid.npz --sensitivity
