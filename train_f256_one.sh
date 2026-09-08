#!/bin/bash
#SBATCH --job-name=f256s
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f256s%a-%A.log
#SBATCH --array=1-4
# One seed per array task. Four seeds in a single 8h job needed ~10-12h on the
# larger candf256 dumps and would have hit the wall the way seedvar did,
# training some seeds and evaluating none. Independent tasks also run in
# parallel instead of in series.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf256
M=/scratch/pmohseni/omr/scorer/f256; mkdir -p "$M"
S=${SLURM_ARRAY_TASK_ID}
T="$M/vel_p8_f256_s$S.pt"
[ -f "$T" ] && { echo "seed $S already present"; exit 0; }
echo "##### fitting f256 seed $S"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 8 --seed $S 2>&1 | tail -3
echo "##### seed $S done"
