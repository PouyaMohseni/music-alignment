#!/bin/bash
#SBATCH --job-name=n1-long-context
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/n1_long_context-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/n1_long_context-%j.log

# N1: replace CB_TA's 1-layer LSTM with a two-tier (fine + compressed) memory
# Transformer temporal core. Motivated by the 2026-07-31 eval sweep: in the
# best model (B1a, 89.2% pct@0.5s) the failing pieces have MEDIAN onset error
# 0.000s but MEAN 1.3-12.4s -- the model is exact most of the time and
# teleports in bursts to visually similar passages (repeats). Deciding "have I
# played this already?" needs comparison against the piece's distant past,
# which a fixed-size recurrent state cannot hold and attention over an
# explicit history can. Visual encoder and FiLM untouched (V-DINOv2's 6.9%
# and the FiLM-replacement results say not to touch those).

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
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/N1_long_context
mkdir -p "$OUT/runs" "$OUT/params"

# Resume from our own latest; on the FIRST run warm-start from B1a's converged
# checkpoint so only the new temporal core trains from scratch.
PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    # MUST be absolute: this script cd's to $REPO/audio_conditioned_unet before
    # invoking python, so a relative path resolves against the wrong directory
    # and train_model.py dies with FileNotFoundError on a checkpoint that does
    # exist. (Exactly what killed jobs 66850715/16/17.)
    B1A_CKPT=$(find /project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B1a_mert_swap/params \
               -name "best_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
               | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$B1A_CKPT" ] && B1A_CKPT=$(readlink -f "$B1A_CKPT")
    if [ -n "$B1A_CKPT" ]; then
        echo "Cold start: warm-starting from B1a best_model.pt -> $B1A_CKPT"
        PARAM_FLAG="--param_path $B1A_CKPT"
    else
        echo "No B1a checkpoint found, training from scratch"
    fi
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid
export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

echo "=== N1: two-tier memory Transformer temporal core (MERT audio, FiLM unchanged) ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_n1_long_context.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag N1_long_context \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
