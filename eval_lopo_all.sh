#!/bin/bash
#SBATCH --job-name=lopoall
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/lopoall_%a-%A.log
# Every configuration under the one fixed protocol, selected on validation.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python extensions/analysis/lopo_all.py --shard ${SLURM_ARRAY_TASK_ID} --num_shards 6
