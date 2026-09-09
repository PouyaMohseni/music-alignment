#!/bin/bash
#SBATCH --job-name=crf
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/crf%a-%A.log
#SBATCH --array=0-1
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/crf; mkdir -p "$M"
S=${SLURM_ARRAY_TASK_ID}
T="$M/crf_s$S.pt"
[ -f "$T" ] && { echo "present"; exit 0; }
python extensions/analysis/train_crf.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --featproj 8 --K 32 --epochs 12 --seed $S 2>&1 | grep -vE "^\s*$"
[ -f "$T" ] || { echo "!!!!! no checkpoint"; exit 1; }
