#!/bin/bash
#SBATCH --job-name=rankdo
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/rankdo-%j.log
# Can `do` defend 94.0? It is the only proxy with signal against room, and it
# now carries backbone features so it can finally rank the feature models.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python -u extensions/analysis/rank_feat_on_do.py
echo ""; echo "Job finished at $(date)"
