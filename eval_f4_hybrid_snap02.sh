#!/bin/bash
#SBATCH --job-name=f4-hybrid-s02
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f4_hybrid_s02-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/f4_hybrid_s02-%j.log

# F4 full run: per-frame hybrid decode (snap_frac=0.2), the winner from the
# 20-piece smoke test -- outright beat original decode on pct@0.5s (83.6%
# vs 82.0%) AND cut mean_err by ~3x (0.476s vs 1.288s). This is the first
# decode variant this session that improves precision AND rescues the
# long-piece drift tail simultaneously, rather than trading one for the
# other (offline_dtw/particle_filter alone did the latter at a cost to the
# former).

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --decoders original,offline_dtw,hybrid_snap --snap_frac 0.2 \
    --split test --out_dir results/f3_ensemble_decode/hybrid_snap_0.2

echo "Job finished at $(date)"
