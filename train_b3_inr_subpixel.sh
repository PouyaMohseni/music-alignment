#!/bin/bash
#SBATCH --job-name=b3-inr-subpixel
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/b3_inr_subpixel-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/b3_inr_subpixel-%j.log

# CB_TA-Ext B3: local sub-pixel INR refinement at decoder_final (the layer
# just before conv_out), on the same data/config as A0. The most novel of
# the extensions -- targets the tile-quantization floor on tight accuracy
# bins (<=0.05s/<=0.1s) that every prior tile/argmax-based decode inherits.
#
# Originally trained from scratch every submission (no warm-start) for a
# clean ablation number -- but that meant a 24h TIMEOUT would silently
# restart from epoch 0 and lose all progress, unlike every other B-series
# script. Added the same warm-start-from-latest-checkpoint pattern as
# train_b2_pitch_aux.sh/train_cpjku_paper_msmd_aug.sh (2026-07-17, after
# job 65370654 timed out at 24h with this gap still present).

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
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B3_inr_subpixel
mkdir -p "$OUT/runs" "$OUT/params"

# Warm-start from the latest checkpoint if a previous run left one (weights
# only -- CPJKU's train_model.py has no true resume, so epoch/optimizer/
# LR-schedule/early-stop state all restart, but training does not start
# from random init). Same pattern as train_b2_pitch_aux.sh.
PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo "Warm-starting from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    echo "No previous checkpoint found, training from scratch"
fi

echo "=== B3: INR sub-pixel refinement (same data/config as A0) ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_b3.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set /scratch/pmohseni/msmd_train_full \
    --val_set   ../data/msmd/msmd_valid \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder CBEncoder \
    --tag B3_inr_subpixel \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
