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
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
S=${SLURM_ARRAY_TASK_ID}
F=/scratch/pmohseni/omr/cand_enc_$ARM
M=/scratch/pmohseni/omr/scorer/enc; mkdir -p "$M"
T="$M/${ARM}_s$S.pt"
[ -f "$T" ] && { echo "present: $T"; exit 0; }
[ -f "$F/valid.npz" ] || { echo "FATAL: no dumps under $F"; exit 1; }
echo "##### $ARM seed=$S dumps=$F"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 64 --seed "$S" --no_tempo 2>&1 \
    | grep -E "BEST|wrote|rror|Traceback" | tail -4
[ -f "$T" ] || { echo "NO CHECKPOINT"; exit 1; }
