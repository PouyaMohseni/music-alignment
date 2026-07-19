#!/bin/bash
#SBATCH --job-name=music-align-v4
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v4_pitch/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v4_pitch/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v4_pitch

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Python: $(which python)"

# v4 pitch-fused full-seq. Frozen precomputed (LoRA-adapted) embeddings + two
# auxiliary pitch heads (BCE vs MIDI) fused into the matching space + sharpened
# localization loss. Trains in ~15-30 min (only a small head). End-to-end at
# inference (pitch heads are internal features, not a symbolic pivot).
EMB_ROOT=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_lora

python -m mymodel.v4_pitch.train \
  --config configs/v4_pitch.yaml \
  data.emb_root=$EMB_ROOT

echo "Training finished at $(date). Running eval..."

CKPT=results/v4_pitch/best_model.pt
if [ ! -f "$CKPT" ]; then
    echo "WARNING: no best_model.pt found -- falling back to latest checkpoint_*.pt"
    CKPT=$(ls results/v4_pitch/checkpoint_*.pt 2>/dev/null | sort | tail -1)
fi
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found after training"; exit 1
fi
echo "Evaluating: $CKPT"

echo "=== Eval v4_pitch on test split ==="
python -m mymodel.v4_pitch.eval \
    --checkpoint $CKPT \
    --config     configs/v4_pitch.yaml \
    --split      test \
    --emb_root   $EMB_ROOT \
    --out_dir    results/v4_pitch/eval

echo "Job finished at $(date)"
