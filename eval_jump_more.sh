#!/bin/bash
#SBATCH --job-name=evaljump2
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/evaljump2_%a-%A.log
# Discontinuity recovery on the two dumps the first pass did not cover:
# the strongest perception model on the real recordings, and the 94-piece
# synthetic benchmark. Each with and without a silent break before the jump.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
case ${SLURM_ARRAY_TASK_ID} in
  0) O=/scratch/pmohseni/omr/jump_cyolo_sb_a_room_gap8; C=/scratch/pmohseni/omr/scorer/extra/cyolo_sb_a_s0.pt ;;
  1) O=/scratch/pmohseni/omr/jump_cyolo_sb_a_room_gap0; C=/scratch/pmohseni/omr/scorer/extra/cyolo_sb_a_s0.pt ;;
  2) O=/scratch/pmohseni/omr/jump_cyolo_sb_set1_gap8;   C=/scratch/pmohseni/omr/scorer/nbr/nbrp64_s0.pt ;;
  3) O=/scratch/pmohseni/omr/jump_cyolo_sb_set1_gap0;   C=/scratch/pmohseni/omr/scorer/nbr/nbrp64_s0.pt ;;
esac
echo "##### $(basename $O)  $(date)"
python extensions/analysis/jump_eval.py --dump "$O/cand.npz" --side "$O/side.npz" \
    --ckpt "$C" --modes none break conf beam --argmax
