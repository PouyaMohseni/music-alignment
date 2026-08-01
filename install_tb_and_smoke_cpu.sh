#!/bin/bash
#SBATCH --job-name=tb-smoke-cyolo
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/tb_smoke_cyolo-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/tb_smoke_cyolo-%j.log
# train.py imports tensorboard at MODULE level, so --no_log does not avoid it.
# Safe to install now: the SOTA eval that was using this venv has finished.
# Installed under the same constraints file so numpy cannot drift >=1.24 and
# re-break madmom (that exact drift cost a rebuild earlier).
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
pip install --no-index -c /scratch/pmohseni/venv_cyolo/constraints.txt tensorboard 2>&1 | tail -2
python -c "import numpy,tensorboard,madmom,cv2; print('numpy',numpy.__version__,'| tb ok | madmom',madmom.__version__)" \
  || { echo "FATAL: venv broken after tensorboard install"; exit 1; }
echo "Job finished at $(date)"
