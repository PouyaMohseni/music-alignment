#!/bin/bash
#SBATCH --job-name=eval-v8
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v8-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v8-%j.log

echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CKPT=$(ls results/v8_henkel_repro/checkpoint_*.pt 2>/dev/null | sort | tail -1)

if [ -z "$CKPT" ]; then
  echo "ERROR: no v8_henkel_repro checkpoint found"; exit 1
fi
echo "Evaluating: $CKPT"

python -m mymodel.v8_henkel_repro.eval \
  --checkpoint $CKPT \
  --config configs/v8_henkel_repro.yaml \
  --split test \
  --processed $PROC \
  --out_dir results/v8_henkel_repro/eval

echo "Job finished at $(date)"
