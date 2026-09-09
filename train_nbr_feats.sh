#!/bin/bash
#SBATCH --job-name=nbrfeat
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/nbrfeat%a-%A.log
#SBATCH --array=0-3
# Nine per-candidate features asking "is this a real notehead" rather than "is
# this where I expect one": neighbourhood density, spacing to either side,
# reading-order rank, box size against the frame median, aspect, and position
# through the predicted bar.
#
# Motivated by the failure analysis rather than guessed: what betrays a
# lock-loss episode is that the chosen box has LOW OBJECTNESS (AUC 0.805) --
# the tracker is somewhere with no music and takes whatever box exists there --
# while position says almost nothing (0.558). log_ncand already counts
# candidates but is identical for every candidate in a frame, so it cannot
# rank. These are its per-candidate versions.
#
# Four seeds, because today established that one draw of this config spans two
# points on room and a single checkpoint proves nothing.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/nbr; mkdir -p "$M"
S=${SLURM_ARRAY_TASK_ID}
T="$M/nbr_s$S.pt"
[ -f "$T" ] && { echo "seed $S present"; exit 0; }
echo "##### nbr features, seed $S"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 8 --seed $S 2>&1 | tail -3
echo "##### seed $S done"
