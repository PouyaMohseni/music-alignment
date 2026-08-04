#!/bin/bash
#SBATCH --job-name=r2a-channel-aug
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/r2a_channel_aug-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/r2a_channel_aug-%j.log

# R2a -- train the MERT path against a TIME-CONSTANT per-dimension channel
# perturbation, so the model cannot rely on absolute per-dimension levels that
# a different piano and room will not reproduce. This is the zero-precompute
# proxy for R2 (waveform augmentation + MERT re-encode, 6615 renders); running
# both says whether that cost buys anything a feature-space proxy does not.
#
# Warm-starts from B1a_mert_swap (38.5 room), the CLEAN MERT base, so the delta
# is attributable to the augmentation alone rather than confounded with the
# pitch auxiliary loss.

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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/R2a_channel_aug
mkdir -p "$OUT/runs" "$OUT/params"

# Resume from our own latest; on the FIRST run warm-start from B1a_mert_swap.
# No new parameters are introduced -- the augmentation is a forward-pass
# transform -- so this is a plain fine-tune of the converged base.
#
# Paths MUST be absolute -- this script cd's to $REPO/audio_conditioned_unet
# before invoking python, so a relative checkpoint path resolves against the
# wrong directory and train_model.py dies with FileNotFoundError on a
# checkpoint that does exist (this is what killed jobs 66850715/16/17).
PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    LATEST_CKPT=$(readlink -f "$LATEST_CKPT")
    echo "Resuming from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    BASE=$(find /project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B1a_mert_swap/params \
           -name "best_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$BASE" ] && BASE=$(readlink -f "$BASE")
    if [ -n "$BASE" ] && [ -f "$BASE" ]; then
        echo "Cold start: warm-starting from B1a_mert_swap -> $BASE"
        PARAM_FLAG="--param_path $BASE"
    else
        echo "FATAL: no B1a_mert_swap checkpoint to warm-start from"; exit 1
    fi
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid
export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== R2a: MERT + time-constant channel augmentation ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_r2a_channel_aug.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag R2a_channel_aug \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
