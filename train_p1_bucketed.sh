#!/bin/bash
#SBATCH --job-name=p1-bucketed
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/p1_bucketed-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/p1_bucketed-%j.log

# P1 -- replace the dense soft-Dice heatmap objective with a bucketed softmax
# over x position.  See extensions/heads/bucketed_softmax.py for the full
# rationale; the short version:
#
#   * MM-Loc 58.5 vs CUNet 22.4 on room -- same paper, same lab, same data,
#     same audio tower.  A 36-point swing whose only material difference is the
#     output parameterisation.
#   * AMT recovers 91-95% of onsets from the very recordings we score 56.6 on,
#     so the information survives the room and the failure is representational.
#   * Real-IR augmentation bought CYOLO (detection) +25.2 but bought our
#     heatmap model only +11 from the same IR bank -- consistent with the
#     output layer, not the input distribution, being the binding constraint.
#
# ZERO NEW PARAMETERS: conv_out is already a 1x1 conv to a single logit
# channel, so we marginalise its logits over height and softmax over x.  The
# checkpoint stays shape-compatible with the warm start, and this run differs
# from R2r_realir in LOSS AND DECODE ONLY.
#
#   usage: sbatch train_p1_bucketed.sh [pool] [dice_weight]
#          pool        = logsumexp (default) | mean | max
#          dice_weight = 0 (default, pure reparameterisation) | >0 hybrid

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi | head -12

POOL=${1:-logsumexp}
DICE_W=${2:-0}
TAG=P1_bucketed_${POOL}$([ "$DICE_W" != "0" ] && echo "_dice${DICE_W}")

cd /project/def-ichiro/pmohseni/music-alignment
if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/train_model.py ]; then
    git submodule update --init third_party/cpjku_unet || true
fi
module load gcc opencv
# venv_cpjku310, NOT .venv -- audio_conditioned_unet/utils.py:12 imports madmom
# at module level and .venv has no madmom, so every import of the cpjku package
# dies there (this is the second thing that killed job 551057).  R2/B* all use
# this venv; P1 warm-starts from an R2 checkpoint and must match it exactly.
source /scratch/pmohseni/venv_cpjku310/bin/activate

OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/${TAG}
mkdir -p "$OUT/params" "$OUT/runs"

# Absolute paths only -- this script cd's into audio_conditioned_unet before
# invoking python, so a relative checkpoint path resolves against the wrong
# directory (this is what killed 66850715/16/17).
PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    LATEST_CKPT=$(readlink -f "$LATEST_CKPT")
    echo "Resuming from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    # Warm-start from R2r_realir -- our best real-audio model (56.6 room).  P1
    # therefore measures the output reparameterisation ON TOP of IR
    # augmentation, rather than re-deriving a gain R2r already banked.
    BASE=$(find /project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/R2r_realir/params \
           -name "best_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$BASE" ] && BASE=$(readlink -f "$BASE")
    if [ -n "$BASE" ] && [ -f "$BASE" ]; then
        echo "Cold start: warm-starting from R2r_realir -> $BASE"
        PARAM_FLAG="--param_path $BASE"
    else
        echo "FATAL: no R2r_realir checkpoint to warm-start from"; exit 1
    fi
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid
export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"
export P1_POOL="$POOL"
export P1_DICE_WEIGHT="$DICE_W"

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== ${TAG}: bucketed-softmax position objective (pool=$POOL dice_w=$DICE_W) ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_p1_bucketed.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag ${TAG} \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
