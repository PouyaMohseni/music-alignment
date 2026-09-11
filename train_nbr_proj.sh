#!/bin/bash
#SBATCH --job-name=nbrproj
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/nbrproj%a-%A.log
#SBATCH --array=0-7
# nbr proj 64 wins BOTH sets on 2 seeds (room 94.54, held-out 90.08), but proj
# 32 comes out WORSE than proj 8 (91.19 vs 93.17). A width effect that goes
# down then up is the signature of n=2 noise, not of a real optimum at 64. Four
# more seeds of each so both means rest on six draws before anything is chosen.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/nbr
I=${SLURM_ARRAY_TASK_ID}
P=$([ "$I" -lt 4 ] && echo 64 || echo 32); S=$(( (I % 4) + 2 ))
T="$M/nbrp${P}_s$S.pt"
[ -f "$T" ] && { echo "present: $T"; exit 0; }
echo "##### nbr featproj=$P seed=$S"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj $P --seed $S --no_tempo 2>&1 | tail -4
[ -f "$T" ] || { echo "!!!!! no checkpoint"; exit 1; }
