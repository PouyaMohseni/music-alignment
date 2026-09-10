#!/bin/bash
#SBATCH --job-name=nbrmore
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/nbrmore%a-%A.log
#SBATCH --array=0-7
# nbr is the only configuration better than the shipped one on BOTH sets:
# room 93.38 against 92.20 as config means, held-out 89.60 against 89.27, and
# all four of its seeds beat the featureless baseline with intervals excluding
# zero. But +1.18 over 4-5 seeds is about 1.2 pooled sd -- suggestive, not
# resolved.
#
# Tasks 0-3 add four more seeds of the same config so the mean has 8 draws
# behind it. Tasks 4-7 widen the per-candidate projection, which has never been
# swept WITH the neighbourhood features present; 8 was chosen when there were
# 24 features, and there are 33 now.
#
# Selection stays on held-out. Room is read once, through the harness, for the
# winner only.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/nbr; mkdir -p "$M"
I=${SLURM_ARRAY_TASK_ID}
if [ "$I" -lt 4 ]; then P=8;  S=$((I + 4)); T="$M/nbr_s$S.pt"
else                   P=$([ "$I" -lt 6 ] && echo 32 || echo 64)
                       S=$((I % 2));       T="$M/nbrp${P}_s$S.pt"; fi
[ -f "$T" ] && { echo "present: $T"; exit 0; }
echo "##### nbr featproj=$P seed=$S"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj $P --seed $S --no_tempo 2>&1 | tail -4
[ -f "$T" ] || { echo "!!!!! no checkpoint"; exit 1; }
