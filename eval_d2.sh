#!/bin/bash
#SBATCH --job-name=eval-d2
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_d2-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_d2-%j.log

# Interim eval of D2's current best_model.pt while training is still ongoing.
# eval.py is D1's, re-exported unchanged by D2 (see D2.md) -- MIDI never
# touches this. band_frac=0.05 default, matches D1's measured-best setting.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

CKPT=results/d2_midi_privileged/best_model.pt

echo "=== offline DTW (banded, band_frac=0.05) ==="
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged.yaml --checkpoint "$CKPT" --split test --band_frac 0.05

echo ""
echo "=== causal OLTW (online) ==="
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged.yaml --checkpoint "$CKPT" --split test --online

echo "Job finished at $(date)"
