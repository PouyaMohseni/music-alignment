#!/bin/bash
#SBATCH --job-name=idxsel
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/idxsel-%j.log
# Does the note-index prior stack on the shipped 91.4 selector, or has the
# selector already learned it? The index is worth +0.7 on the hand decoder
# alone; the selector already sees per-candidate displacement, so it may be
# redundant. Swept over the blend so we can see whether the two interact.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python -u extensions/analysis/run_index_selector.py
echo ""; echo "Job finished at $(date)"
