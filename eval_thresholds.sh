#!/bin/bash
#SBATCH --job-name=evalth
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/evalth_%a-%A.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
M=/scratch/pmohseni/omr/scorer; O=/scratch/pmohseni/omr
case ${SLURM_ARRAY_TASK_ID} in
  0) D=$O/cand_cyolo/room.npz;      C="$M/extra/cyolo_s[0-9].pt";      L="CYOLO + CANDOR" ;;
  1) D=$O/candf256/room.npz;        C="$M/nbr/nbrp64_s[0-5].pt";       L="CYOLO-SB + CANDOR" ;;
  2) D=$O/cand_cyolo_sb_a/room.npz; C="$M/extra/cyolo_sb_a_s[0-9].pt"; L="CYOLO-SB+A + CANDOR" ;;
esac
python extensions/analysis/lopo_thresholds.py --dump "$D" --ckpt "$C" --label "$L"
