#!/bin/bash
#SBATCH --job-name=smoke-f7-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f7_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f7_cpu-%j.log

# F7 smoke test: confidence-gated single-model trust, targeting the
# near-perfect-precision gap found in the cross-model distribution
# comparison (our F4 ensemble trails the official model specifically on
# easy/already-well-tracked pieces, not on catastrophic failures).
#
# Routed through CPU (per_onset_error_diagnostic.py's proven workaround)
# since the A100 GPU partition has been fully saturated for 8+ hours with
# zero movement on any job. This script auto-falls back to CPU when no GPU
# is available/requested.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 30 pieces -- large enough for a meaningful read, small enough to fit a
# CPU smoke test in a few hours given ~4-5min/piece observed throughput
# for this same 3-model ensemble in the per-onset diagnostic.
python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --decoders original,hybrid_snap,conf_trust,conf_trust_snap \
    --snap_frac 0.2 --conf_trust_window 5 --conf_trust_margin 0.15 --conf_trust_max_shift 5.0 \
    --split test --limit 30 --out_dir results/f3_smoke/conf_trust --device cpu

echo "Job finished at $(date)"
