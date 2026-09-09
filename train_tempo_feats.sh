#!/bin/bash
#SBATCH --job-name=tempofeat
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/tempofeat%a-%A.log
#SBATCH --array=0-3
# Tempo as a FEATURE rather than a decode rule.
#
# Tracking tempo inside the prior lost outright (room 93.42 -> 90.62): step
# sizes vary 6x within a piece, so centring a Gaussian on the tracked median
# penalises the many short steps, and the conservative fwd_px=6.0 wins by
# leaving the call to objectness.
#
# That is the same shape as the velocity result, where a constant-velocity
# DECODE failed and the identical quantity as a FEATURE worked -- a feature can
# be weighted per frame, a prior commits on every one. So this offers the
# tracked tempo to the model instead of imposing it. It differs from d_extrap
# in having MEMORY: an EMA over the piece rather than a two-point estimate,
# which is what lets it survive the frames either side of a bad step.
#
# Also carries the nine neighbourhood features, so this is the full 37 against
# vel_p8's 24.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/tempo; mkdir -p "$M"
S=${SLURM_ARRAY_TASK_ID}
T="$M/tempo_s$S.pt"
[ -f "$T" ] && { echo "seed $S present"; exit 0; }
echo "##### tempo+nbr features, seed $S"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 8 --seed $S 2>&1 | tail -25
[ -f "$T" ] || { echo "!!!!! seed $S produced no checkpoint"; exit 1; }
echo "##### seed $S done"
