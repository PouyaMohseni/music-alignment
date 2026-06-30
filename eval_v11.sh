#!/bin/bash
#SBATCH --job-name=eval-v11
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v11-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v11-%j.log

# Evaluate v11 (full-strip BPTT) on MSMD test split.
# Reports pct_within_0.5s — comparable to Henkel et al. ISMIR 2020 Table 3.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

CKPT=results/v11_cpjku_fullstrip/best_model.pt
CFG=configs/v11_cpjku_fullstrip.yaml

echo "Checkpoint: $CKPT"
echo "Config:     $CFG"
echo ""

python -m mymodel.v11_cpjku_fullstrip.eval \
    --checkpoint "$CKPT" \
    --config     "$CFG" \
    --split      test \
    --processed  data/MSMD/processed

echo ""
echo "Job finished at $(date)"
