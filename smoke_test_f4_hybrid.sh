#!/bin/bash
#SBATCH --job-name=smoke-f4-hybrid
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f4_hybrid-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f4_hybrid-%j.log

# F4: per-frame hybrid decode -- default to original threshold+CoM, snap to
# the offline-DTW path only on frames where they disagree by more than
# snap_frac*W_sc (signals drift). F3's error analysis showed offline_dtw
# rescues catastrophic long-piece failures (e.g. Chopin Op.9: 32.08s->1.27s
# mean err) but a pure per-piece switch on duration doesn't move pct@0.5s
# (stays ~72%, since DTW's fine-grained precision is worse everywhere).
# Testing whether a per-FRAME hybrid preserves precision on well-tracked
# passages while still catching derailment.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

for SNAP in 0.05 0.1 0.2; do
  echo "=== snap_frac=$SNAP ==="
  python -m mymodel.f3_ensemble_decode.eval \
      --models v13,v14,v15 --decoders original,offline_dtw,hybrid_snap --snap_frac $SNAP \
      --split test --limit 20 --out_dir results/f3_smoke/hybrid_snap_$SNAP
done

echo "Job finished at $(date)"
