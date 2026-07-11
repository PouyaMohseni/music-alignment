#!/bin/bash
#SBATCH --job-name=d2-multitempo
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/d2_multitempo-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/d2_multitempo-%j.log

# E4: D2 extended to multi-tempo whole-piece training data (750/1000/1250),
# now that the full-scale precompute has finished (746 tempo-variant files,
# 0 failures -- see multitempo_wholepiece-*.log). Tests whether more training
# data compounds with D2's already-strong per-signal training quality.

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/d2_midi_privileged_multitempo
if [ ! -f third_party/cpjku_unet/network.py ]; then
    git submodule update --init third_party/cpjku_unet || true
fi
git -C third_party/cpjku_unet checkout ismir-2020

module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

RESUME_FLAG=""
if [ -f results/d2_midi_privileged_multitempo/checkpoint_latest.pt ]; then
    echo "Found existing checkpoint -- resuming."
    RESUME_FLAG="--resume"
fi

python -m mymodel.d2_midi_privileged.train_multitempo \
    --config configs/d2_midi_privileged_multitempo.yaml \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval (offline DTW, banded)..."
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged_multitempo.yaml \
    --checkpoint results/d2_midi_privileged_multitempo/best_model.pt \
    --split test --decoder dtw --band_frac 0.05

echo ""
echo "=== also evaluating particle-filter causal decode ==="
python -m mymodel.d2_midi_privileged.eval \
    --config configs/d2_midi_privileged_multitempo.yaml \
    --checkpoint results/d2_midi_privileged_multitempo/best_model.pt \
    --split test --decoder particle_filter --pf_process_noise_std 3.0 --pf_init_std 2.0

echo "Job finished at $(date)"
