#!/bin/bash
#SBATCH --job-name=reccost
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/reccost-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 scipy-stack opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python extensions/analysis/recovery_cost.py \
    --dump /scratch/pmohseni/omr/cand_cyolo_sb_a/room.npz \
    --ckpt "/scratch/pmohseni/omr/scorer/extra/cyolo_sb_a_s[0-9].pt" \
    --modes none conf beam --label "CYOLO-SB+A, unspliced recordings"
