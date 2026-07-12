#!/bin/bash
#SBATCH --job-name=d3-hybrid
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/d3_hybrid-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/d3_hybrid-%j.log

# D3: D2's architecture/training/decode, with v13's trained MERTProjector
# warm-started as the audio tower instead of D1/D2's own small conv tower.
# See mymodel/d3_hybrid/model.py + configs/d3_hybrid.yaml.
# Prereq: same MERT precompute as D1/D2 (already done) + v13's checkpoint
# (already trained, /scratch/pmohseni/results/v13_mert_linear/best_model.pt).

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/d3_hybrid
SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

RESUME_FLAG=""
if [ -f results/d3_hybrid/checkpoint_latest.pt ]; then
    echo "Found existing checkpoint -- resuming."
    RESUME_FLAG="--resume"
fi

python -m mymodel.d3_hybrid.train \
    --config configs/d3_hybrid.yaml \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval (offline DTW, banded)..."
python -m mymodel.d3_hybrid.eval \
    --config configs/d3_hybrid.yaml \
    --checkpoint results/d3_hybrid/best_model.pt \
    --split test --decoder dtw --band_frac 0.05

echo ""
echo "=== also evaluating particle-filter causal decode ==="
python -m mymodel.d3_hybrid.eval \
    --config configs/d3_hybrid.yaml \
    --checkpoint results/d3_hybrid/best_model.pt \
    --split test --decoder particle_filter --pf_process_noise_std 3.0 --pf_init_std 2.0

echo "Job finished at $(date)"
