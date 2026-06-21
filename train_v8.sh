#!/bin/bash
#SBATCH --job-name=music-v8
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v8_henkel_repro-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v8_henkel_repro-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v8_henkel_repro

source .venv/bin/activate
# v8 does NOT use HF models — no HF_HUB_OFFLINE needed

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed

# Check audio.wav exists for at least one piece
FIRST=$(python3 -c "import json; ids=json.load(open('$PROC/splits.json'))['train']; print(ids[0])")
if [ ! -f "$PROC/$FIRST/audio.wav" ]; then
  echo "ERROR: audio.wav not found at $PROC/$FIRST/audio.wav"
  echo "Run audio synthesis first:"
  echo "  python -m msmd_prep.run_all --stage synth --processed $PROC"
  exit 1
fi
echo "audio.wav check passed for piece: $FIRST"

python -m mymodel.v8_henkel_repro.train \
  --config configs/v8_henkel_repro.yaml \
  data.processed_root=$PROC

echo "Training finished at $(date). Running eval..."

CKPT=$(ls results/v8_henkel_repro/checkpoint_*.pt 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
  echo "ERROR: no checkpoint found after training"; exit 1
fi
echo "Evaluating: $CKPT"

python -m mymodel.v8_henkel_repro.eval \
  --checkpoint $CKPT \
  --config configs/v8_henkel_repro.yaml \
  --split test \
  --processed $PROC \
  --out_dir results/v8_henkel_repro/eval

echo "Job finished at $(date)"
