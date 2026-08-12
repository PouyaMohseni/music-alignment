#!/bin/bash
#SBATCH --job-name=a4-boundary
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/a4_boundary-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/a4_boundary-%j.log

# boundary-oriented output (BAM-DETR) + coarse staff head, replacing dense soft-Dice
#
# Warm-starts from R2r_realir (56.6 room), so the delta is attributable to this
# change alone rather than re-deriving the MERT tower and real-IR gains.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi | head -12

TAG=${A_TAG:-A4_boundary}
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
# venv_cpjku310, NOT .venv: audio_conditioned_unet/utils.py:12 imports madmom at
# module level and .venv has none, so the whole package fails to import.
source /scratch/pmohseni/venv_cpjku310/bin/activate

OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/${TAG}
mkdir -p "$OUT/params" "$OUT/runs"

# Absolute paths only: this script cd's into audio_conditioned_unet before
# invoking python, so a relative checkpoint path resolves elsewhere.
PARAM_FLAG=""
LATEST=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST" ]; then
    LATEST=$(readlink -f "$LATEST"); echo "Resuming from $LATEST"
    PARAM_FLAG="--param_path $LATEST"
else
    BASE=$(find results/cb_ta_ext/R2r_realir/params -name "best_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$BASE" ] && BASE=$(readlink -f "$BASE")
    [ -f "$BASE" ] || { echo "FATAL: no R2r_realir checkpoint"; exit 1; }
    echo "Cold start: warm-starting from R2r_realir -> $BASE"
    PARAM_FLAG="--param_path $BASE"
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid
export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== ${TAG}: boundary-oriented output (BAM-DETR) + coarse staff head, replacing dense soft-Dice ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_a4_boundary.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag ${TAG} \
    $PARAM_FLAG

echo ""; echo "Training finished at $(date)"
