#!/bin/bash
#SBATCH --job-name=heldout
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/heldout-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python extensions/analysis/heldout_compare.py --snrs 12,6,3,0.5
