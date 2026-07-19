#!/bin/bash
#SBATCH --job-name=mert-dinov2-crossattn
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mert_dinov2_crossattn-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/mert_dinov2_crossattn-%j.log

# The general MERT+DINOv2 cross-attention experiment: MERT audio encoder +
# DINOv2 visual-token neck (residual/bottleneck content pathway, same as
# V-DINOv2) + TokenCrossAttentionFiLM as the fusion mechanism -- MERT's
# audio embedding is the attention QUERY, the raw (un-interpolated) DINOv2
# patch tokens are KEY/VALUE, replacing FiLM's context-blind uniform
# broadcast entirely. This is the "actual new method": both encoders AND
# the fusion mechanism change together, unlike V-DINOv2 (encoder swap only,
# FiLM untouched) or B1a-cross-attention (fusion swap only, encoder
# untouched). Needs the DINOv2 tiled-native precompute
# (scripts/precompute_dinov2_tiled_native.py) to have completed.

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
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/MERT_dinov2_cross_attention
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

export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

echo "=== MERT+DINOv2 cross-attention (MERT query attends over raw DINOv2 patch tokens) ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_mert_dinov2_cross_attention.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag MERT_dinov2_cross_attention \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
