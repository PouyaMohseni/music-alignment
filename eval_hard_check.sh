#!/bin/bash
#SBATCH --job-name=hardchk
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/hardchk-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python -u /tmp/claude-3152518/-lustre06-project-6002780-pmohseni-music-alignment/221a9f7b-1956-4881-9c3f-4fbd1f392674/scratchpad/hard.py
