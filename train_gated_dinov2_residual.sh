#!/bin/bash
#SBATCH --job-name=gated-dinov2-residual
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/gated_dinov2_residual-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/gated_dinov2_residual-%j.log

# Rescue attempt for V-DINOv2 (6.9% pct@0.5s, catastrophic -- a full
# REPLACEMENT of CB_TA's from-scratch conv encoder with DINOv2's much
# coarser tile-grid features). This keeps the original encoder as the
# primary path and adds DINOv2 features only as a zero-initialized additive
# residual (extensions/hooks/gated_dinov2_residual_patch.py) -- confirmed
# via smoke test to be byte-identical to plain CB_TA at initialization, so
# training can only improve on the known-good baseline, never regress below
# it the way full replacement did. Same "start as identity" principle that
# rescued gated FiLM (82.9%) over full-strength FiLM alternatives. Plain
# CBEncoder audio (not MERT) -- isolates this as a pure visual-architecture
# idea, matching V-DINOv2's own discipline.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DINOV2_TILED_ROOT=/scratch/pmohseni/dinov2_emb_tiled_native

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/Gated_dinov2_residual
mkdir -p "$OUT/runs" "$OUT/params"

PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo "Warm-starting from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    echo "No previous checkpoint found, training from scratch"
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid

echo "=== Gated-residual DINOv2 hybrid (plain CB_TA encoder + zero-init gated DINOv2 residual) ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_gated_dinov2_residual.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder CBEncoder \
    --tag Gated_dinov2_residual \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
