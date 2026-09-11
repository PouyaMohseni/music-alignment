#!/bin/bash
#SBATCH --job-name=lopseed
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH --array=0-5
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/lopseed_%a-%A.log
# The leave-one-piece-out protocol applied to every seed of the shipped
# configuration, so the reported number is a mean over draws under the same
# honest selection rather than one checkpoint's.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python extensions/analysis/lopo_tuning.py --macro \
    --ckpt /scratch/pmohseni/omr/scorer/nbr/nbrp64_s${SLURM_ARRAY_TASK_ID}.pt
