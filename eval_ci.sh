#!/bin/bash
#SBATCH --job-name=evalci
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/evalci_%a-%A.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import cv2, scipy, numpy, torch" || { echo "FATAL: env broken"; exit 1; }
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=/scratch/pmohseni/omr/scorer; O=/scratch/pmohseni/omr
case ${SLURM_ARRAY_TASK_ID} in
  0) D=$O/cand_cyolo_sb_a/room.npz; C="$M/extra/cyolo_sb_a_s[0-9].pt"; L="CYOLO-SB+A + CANDOR (headline)" ;;
  1) D=$O/candf256/room.npz;        C="$M/nbr/nbrp64_s[0-5].pt";       L="CYOLO-SB + CANDOR" ;;
esac
python extensions/analysis/lopo_ci.py --dump "$D" --ckpt "$C" --label "$L"
