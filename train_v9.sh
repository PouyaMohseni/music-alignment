#!/bin/bash
#SBATCH --job-name=music-v9
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v9_cpjku-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v9_cpjku-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v9_cpjku

source .venv/bin/activate

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed

FIRST=$(python3 -c "import json; ids=json.load(open('$PROC/splits.json'))['train']; print(ids[0])")
if [ ! -f "$PROC/$FIRST/audio.wav" ]; then
  echo "ERROR: audio.wav not found. Run: python -m msmd_prep.run_all --stage synth --processed $PROC"
  exit 1
fi

python -m mymodel.v9_cpjku.train \
  --config configs/v9_cpjku.yaml \
  data.processed_root=$PROC

echo "Training finished at $(date). Running eval..."

CKPT=$(ls results/v9_cpjku/checkpoint_*.pt 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
  echo "ERROR: no checkpoint found after training"; exit 1
fi
echo "Evaluating: $CKPT"

python -m mymodel.v9_cpjku.eval \
  --checkpoint $CKPT \
  --config configs/v9_cpjku.yaml \
  --split test \
  --processed $PROC \
  --out_dir results/v9_cpjku/eval

echo "Job finished at $(date)"
