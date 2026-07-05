#!/bin/bash
#SBATCH --job-name=eval-v13-ep20
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v13_ep20-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v13_ep20-%j.log

set -euo pipefail
echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

CKPT=/scratch/pmohseni/results/v13_mert_linear/checkpoint_epoch020.pt
CFG=configs/v13_mert_linear.yaml
OUT=/scratch/pmohseni/results/v13_mert_linear/eval_ep20

echo "Checkpoint: $CKPT"
python -m mymodel.v13_mert_unet.eval \
    --checkpoint "$CKPT" \
    --config     "$CFG"  \
    --split      test    \
    --out_dir    "$OUT"

echo "Job finished at $(date)"
