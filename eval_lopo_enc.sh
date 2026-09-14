#!/bin/bash
#SBATCH --job-name=lopoenc
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/lopoenc-%j.log
# Cross-validate every encoder arm that has a complete chain, each against its
# own detector's candidates, under the protocol used for every other row.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 scipy-stack opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python extensions/analysis/lopo_new.py
