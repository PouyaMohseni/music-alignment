#!/bin/bash
#SBATCH --job-name=eval-m1-fixes-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_m1_fixes_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_m1_fixes_cpu-%j.log

# Test two decode-time-only fixes on M1's EXISTING checkpoint (no retraining):
# (1) soft position readout (local-window blend instead of hard Viterbi snap)
# (2) entropy/margin-based repeat-ambiguity tagging (replaces the broken
#     geometric heuristic that only tagged 11/15632 onsets). See
#     mymodel/m1_monotonic/eval.py's module docstring for the full reasoning.
# CPU job (not GPU) -- this is pure inference, and the CPU queue is far less
# contended right now (confirmed pattern this session).

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.m1_monotonic.eval \
    --config configs/m1_monotonic.yaml \
    --checkpoint results/m1_monotonic/best_model.pt \
    --split test --repeat_stratified

echo "Job finished at $(date)"
