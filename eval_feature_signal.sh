#!/bin/bash
#SBATCH --job-name=featsig
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/featsig-%j.log
# Does note-index displacement carry signal the pixel version does not? Decides
# whether an hour of retraining is worth spending, before spending it.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python -u extensions/analysis/feature_signal.py
echo ""; echo "Job finished at $(date)"
