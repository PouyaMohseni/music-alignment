#!/bin/bash
#SBATCH --job-name=d2-midi-privileged
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/d2_midi_privileged-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/d2_midi_privileged-%j.log

# D2: D1's model + MIDI-privileged training (repeat-aware soft CE labels +
# MIDI->audio distillation), MIDI never used at inference. See D2.md.
# Prereq: same MERT precompute as D1 (precompute_mert_trainval.sh) -- already done.

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/d2_midi_privileged
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

RESUME_FLAG=""
if [ -f results/d2_midi_privileged/checkpoint_latest.pt ]; then
    echo "Found existing checkpoint -- resuming."
    RESUME_FLAG="--resume"
fi

python -m mymodel.d2_midi_privileged.train \
    --config configs/d2_midi_privileged.yaml \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval (offline DTW, banded)..."
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged.yaml \
    --checkpoint results/d2_midi_privileged/best_model.pt \
    --split test --band_frac 0.05

echo ""
echo "=== also evaluating causal OLTW decode ==="
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged.yaml \
    --checkpoint results/d2_midi_privileged/best_model.pt \
    --split test --online

echo "Job finished at $(date)"
