#!/bin/bash
#SBATCH --job-name=b1a-mert-swap
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/b1a_mert_swap-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/b1a_mert_swap-%j.log

# CB_TA-Ext B1a: frozen MERT audio-encoder swap into CB_TA's unmodified
# ConditionalUNet, trained on the SAME Zenodo data/config as A0
# (train_cpjku_paper_CB_TA.sh) so the pct@0.5s delta is a clean measure of
# whether a foundation-model audio encoder helps, isolated from any other
# change. Uses CPJKU's own train_model.py unmodified, run via
# extensions/hooks/run_train_with_mert.py which monkey-patches only the
# audio-feature source (precomputed MERT instead of live mel-spectrogram)
# and registers MERTProjector as a selectable --audio_encoder.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
    git submodule update --init third_party/cpjku_unet || true
fi
git -C third_party/cpjku_unet checkout ismir-2020

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B1a_mert_swap
mkdir -p "$OUT/runs" "$OUT/params"

# Warm-start from the latest checkpoint if a previous run left one (weights
# only -- CPJKU's train_model.py has no true resume, so epoch/optimizer/
# LR-schedule/early-stop state all restart, but training does not start
# from random init). Same pattern as train_cpjku_paper_msmd_aug.sh.
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
VAL_SET=../data/msmd/msmd_valid   # relative to $REPO/audio_conditioned_unet, matches A0's own invocation

export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

echo "=== B1a: MERT audio-encoder swap (same data/config as A0) ==="
echo "train:  $TRAIN_SET  (945 pages x 7 tempi = 6615 pairs)"
echo "val:    $VAL_SET  (28 pieces, all complete)"
echo "config: msmd_aug.yaml (tempo augmentation: 500/750/950/1000/1050/1250/1500)"
echo "encoder: MERTProjector (frozen MERT-v1-95M + Linear(768,256)+ELU+Linear(256,32)) + LSTM + FiLM layers 2-8"
echo ""

cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_with_mert.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag B1a_mert_swap \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
echo "Best model: $OUT/params/*_B1a_mert_swap/best_model.pt"
