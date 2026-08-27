#!/bin/bash
#SBATCH --job-name=b1a-gated-cross-attn
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/b1a_gated_cross_attention-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/b1a_gated_cross_attention-%j.log

# Isolates the confound between B1a-cross-attention (71.1% pct@0.5s) and
# B1a-gated-film (82.9%), both eval'd 2026-07-27/28 (see task #40): same
# SpatialCrossAttentionFiLM mechanism as B1a-cross-attention, but blended in
# via an AdaLN-Zero-style zero-initialized gate exactly like B1a-gated-film
# (extensions/heads/gated_cross_attention_film.py), so training starts as
# pure identity (no audio effect) instead of full-strength untrained
# cross-attention from step one. Tests whether the mechanism itself is the
# problem, or just the lack of gated-film's stabilization trick.

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

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B1a_gated_cross_attention
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

echo "=== B1a + gated cross-attention FiLM (zero-init gate blends in cross-attention modulation) ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_b1a_gated_cross_attention.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag B1a_gated_cross_attention \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
