#!/bin/bash
#SBATCH --job-name=f4-hybrid-s03
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f4_hybrid_s03-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/f4_hybrid_s03-%j.log

# F4 full run: per-frame hybrid decode (snap_frac=0.3) -- testing whether
# the pct@0.5s improvement continues past snap_frac=0.2 (smoke test showed
# a monotonic climb 0.05->0.1->0.2: 81.1%->82.5%->83.6% on 20 pieces) or
# plateaus/reverses.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --decoders original,offline_dtw,hybrid_snap --snap_frac 0.3 \
    --split test --out_dir results/f3_ensemble_decode/hybrid_snap_0.3

echo "Job finished at $(date)"
