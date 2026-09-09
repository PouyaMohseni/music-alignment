#!/bin/bash
#SBATCH --job-name=pitchsc
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/pitchsc%a-%A.log
#SBATCH --array=0-3
# STAGE 2: the pitch agreement as a scorer feature.
#
# Stage 1 trains two heads -- score head (image features -> the pitch of the
# notehead a box sits on) and audio head (z -> the pitches sounding) -- with
# MIDI as supervision. Neither reads anything symbolic at inference, which is
# the constraint: the method has to work on music with no reference.
#
# The feature is their AGREEMENT, four columns (37..40). Agreement alone is the
# only history-independent evidence in the set, and 76% of the remaining error
# is the tracker inheriting its own mistakes, so every other feature is
# computed against a previous position that may already be wrong.
#
# Four seeds, because a single draw of a config spans two points on room.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
H=/scratch/pmohseni/omr/scorer/pitch/heads_s0.pt
M=/scratch/pmohseni/omr/scorer/pitchsc; mkdir -p "$M"
[ -f "$H" ] || { echo "!!!!! pitch heads missing at $H (stage 1 must finish)"; exit 1; }
S=${SLURM_ARRAY_TASK_ID}
T="$M/pitchsc_s$S.pt"
[ -f "$T" ] && { echo "present"; exit 0; }
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 8 --seed $S --pitch_heads "$H" 2>&1 | grep -vE "^\s*$" | tail -20
[ -f "$T" ] || { echo "!!!!! no checkpoint"; exit 1; }
