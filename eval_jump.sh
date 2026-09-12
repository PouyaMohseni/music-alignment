#!/bin/bash
#SBATCH --job-name=evaljump
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/evaljump_%a-%A.log
#SBATCH --array=0-1
# Jump recovery on the spliced room take: argmax, then the shipped decoder with
# no recovery, CODA's break mode, and the two that fire on the tracker's own
# disagreement.  task 0 = 8-frame silence before each jump, task 1 = none.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
TAG=$([ "${SLURM_ARRAY_TASK_ID}" = 0 ] && echo gap8 || echo gap0)
O=/scratch/pmohseni/omr/jump_cyolo_sb_room_$TAG
echo "##### $TAG  $(date)"
python extensions/analysis/jump_eval.py \
    --dump "$O/cand.npz" --side "$O/side.npz" \
    --ckpt /scratch/pmohseni/omr/scorer/nbr/nbrp64_s0.pt \
    --modes none break conf beam --argmax
