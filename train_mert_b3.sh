#!/bin/bash
#SBATCH --job-name=mert-b3
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mert_b3-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/mert_b3-%j.log

# MERT audio-encoder swap (frozen, precomputed embeddings) COMBINED with
# B3's INR sub-pixel refinement auxiliary loss. Warm-started from B1a's OWN
# converged checkpoint (88.9% pct@0.5s, the project's best result as of
# 2026-07-17) on the FIRST run only -- B3's own docs note the refiner's
# coarse-peak input is meaningless noise until the base heatmap is already
# reasonably localized, so starting from a converged base (not scratch)
# matches how the original (CBEncoder) B3 was designed to be used.

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
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/MERT_B3_inr_subpixel
B1A_BEST=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B1a_mert_swap/params/20260716_105522_B1a_mert_swap/best_model.pt
mkdir -p "$OUT/runs" "$OUT/params"

PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo "Warm-starting from own prior checkpoint: $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
elif [ -f "$B1A_BEST" ]; then
    echo "First run -- warm-starting from B1a's converged checkpoint: $B1A_BEST"
    PARAM_FLAG="--param_path $B1A_BEST"
else
    echo "No B1a checkpoint found either, training from scratch"
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid
export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

echo "=== MERT+B3: frozen MERT audio encoder + INR sub-pixel refinement ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_mert_b3.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag MERT_B3_inr_subpixel \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
