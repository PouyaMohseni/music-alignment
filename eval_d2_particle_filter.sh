#!/bin/bash
#SBATCH --job-name=eval-d2-pf
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_d2_pf-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_d2_pf-%j.log

# Full 94-piece eval of D2's checkpoint with the new particle-filter causal
# decoder (process_noise_std=3.0/init_std=2.0, tuned on 3 pieces -- this is
# the definitive full-test-set number). Also re-runs offline DTW and greedy
# OLTW on the same checkpoint for a clean 3-way comparison in one log.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
CKPT=results/d2_midi_privileged/best_model.pt

echo "=== offline DTW (band_frac=0.05) ==="
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged.yaml --checkpoint "$CKPT" --split test --decoder dtw --band_frac 0.05

echo ""
echo "=== greedy OLTW (causal) ==="
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged.yaml --checkpoint "$CKPT" --split test --decoder oltw

echo ""
echo "=== particle filter (causal, process_noise_std=3.0, init_std=2.0) ==="
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged.yaml --checkpoint "$CKPT" --split test --decoder particle_filter \
    --pf_process_noise_std 3.0 --pf_init_std 2.0

echo "Job finished at $(date)"
