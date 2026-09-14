#!/bin/bash
#SBATCH --job-name=evalthenc
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/evalthenc_%a-%A.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import cv2, scipy, numpy, torch" || { echo "FATAL: env broken"; exit 1; }
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARMS=(lstm cnnmamba mamba)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
python extensions/analysis/lopo_thresholds.py \
    --dump /scratch/pmohseni/omr/cand_enc_$A/room.npz \
    --ckpt "/scratch/pmohseni/omr/scorer/enc/${A}_s[0-9].pt" \
    --label "CANDOR on our 24h $A detector"
