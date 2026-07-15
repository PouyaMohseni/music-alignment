#!/bin/bash
#SBATCH --job-name=f7-conf-trust-full
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f7_conf_trust_full-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/f7_conf_trust_full-%j.log

# F7 full 94-piece run. Smoke test (30 pieces) showed a small but consistent
# positive signal: conf_trust_snap beat hybrid_snap by +0.3pp pct@0.5s
# (83.1% vs 82.8%) with near-identical mean_err -- not dramatic, but in the
# right direction and not the kind of regression that sank max-fusion/F5.
# 30 pieces is too small to trust a 0.3pp delta, so confirming at full scale.
# Routed through CPU given the A100 partition has been fully saturated for
# 8+ hours; ~6.4 min/piece observed on the smoke test extrapolates to ~10h
# for 94 pieces, still likely faster than waiting for a GPU slot.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --decoders original,hybrid_snap,conf_trust,conf_trust_snap \
    --snap_frac 0.2 --conf_trust_window 5 --conf_trust_margin 0.15 --conf_trust_max_shift 5.0 \
    --split test --out_dir results/f3_ensemble_decode/conf_trust_full --device cpu

echo "Job finished at $(date)"
