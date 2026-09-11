#!/bin/bash
#SBATCH --job-name=ablenc
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/ablenc_%a-%A.log
# One cell of the encoder ablation (audio encoder x image encoder), seed = array index.
#
#   sbatch --array=0-2 train_encoder_ablation.sh h1|dino|h1dino
#
#   h1      MERT-audio detector,  its own backbone features
#   dino    cyolo_sb,             DINOv2 features
#   h1dino  MERT-audio detector,  DINOv2 features
#   (cyolo_sb with its own backbone features is the shipped nbrp64 config)
#
# The shipped recipe unchanged -- 33 hand features + 128-dim image features,
# projection 64, no tempo -- retrained on each cell's own candidates.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CELL=${1:?usage: train_encoder_ablation.sh h1|dino|h1dino}
S=${SLURM_ARRAY_TASK_ID}
case $CELL in
  h1)     F=/scratch/pmohseni/omr/candh1 ;;
  dino)   F=/scratch/pmohseni/omr/candf_dino ;;
  h1dino) F=/scratch/pmohseni/omr/candh1_dino ;;
  *) echo "unknown cell $CELL"; exit 1 ;;
esac
M=/scratch/pmohseni/omr/scorer/abl; mkdir -p "$M"
T="$M/${CELL}_s$S.pt"
[ -f "$T" ] && { echo "present: $T"; exit 0; }
echo "##### $CELL seed=$S  dumps=$F"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 64 --seed "$S" --no_tempo 2>&1 | tail -4
[ -f "$T" ] || { echo "NO CHECKPOINT"; exit 1; }
