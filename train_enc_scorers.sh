#!/bin/bash
#SBATCH --job-name=trenc
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/trenc_%x_%a-%A.log
# The shipped decision-model recipe on a retrained encoder arm's own proposals.
#   sbatch train_enc_scorers.sh lstm|mamba|cnnmamba|dinov2|cnn
set -uo pipefail
ARM=${1:?usage: train_enc_scorers.sh <arm>}
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 scipy-stack opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import cv2, scipy, numpy, torch" || { echo "FATAL: env broken on $(hostname) -- cv2/scipy/torch not importable"; exit 1; }
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
S=${SLURM_ARRAY_TASK_ID}
F=/scratch/pmohseni/omr/cand_enc_$ARM
M=/scratch/pmohseni/omr/scorer/enc; mkdir -p "$M"
T="$M/${ARM}_s$S.pt"
[ -f "$T" ] && { echo "present: $T"; exit 0; }
# Refuse to train on a partial dump. Twice now a scorer has run to completion
# on a dump still being written, producing a checkpoint fitted on a fifth of
# the data whose numbers looked entirely plausible. The dump is seven files:
# room, valid and five training shards.
NEED="room valid train_c0 train_c1 train_c2 train_c3 train_c4"
MISSING=""
for f in $NEED; do [ -s "$F/$f.npz" ] || MISSING="$MISSING $f"; done
[ -z "$MISSING" ] || { echo "FATAL: incomplete dump in $F, missing:$MISSING"; exit 1; }
# and refuse if any of them was written in the last two minutes, which means
# the dump job is probably still going
RECENT=$(find "$F" -name '*.npz' -newermt '-2 minutes' | wc -l)
[ "$RECENT" -eq 0 ] || { echo "FATAL: $F written within the last 2 min, still in flight"; exit 1; }
echo "dump complete: $(ls "$F"/*.npz | wc -l) files, $(du -sh "$F" | cut -f1)"
echo "##### $ARM seed=$S dumps=$F"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 64 --seed "$S" --no_tempo 2>&1 \
    | grep -E "BEST|wrote|rror|Traceback" | tail -4
[ -f "$T" ] || { echo "NO CHECKPOINT"; exit 1; }
