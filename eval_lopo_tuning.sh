#!/bin/bash
#SBATCH --job-name=lopo
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/lopo-%j.log
# Leave-one-piece-out selection of the prior constants and the blend weight on
# the real recordings (extensions/analysis/lopo_tuning.py).
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python extensions/analysis/lopo_tuning.py
