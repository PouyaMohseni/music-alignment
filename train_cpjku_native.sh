#!/bin/bash
#SBATCH --job-name=cpjku-native-train
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_native_train-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_native_train-%j.log

# Train CPJKU CB_TA architecture from scratch using their unmodified
# train_model.py on our MSMD data.  No adapter, no patches.
#
# Prerequisites:
#   sbatch setup_cpjku310.sh          # create .venv_cpjku310
#   sbatch eval_cpjku_native.sh       # run data conversion first (or run conversion step below)

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/cpjku_native

module load gcc opencv python/3.10
source /scratch/pmohseni/venv_cpjku310/bin/activate

if [ ! -f third_party/cpjku_unet/network.py ]; then
    git submodule update --init third_party/cpjku_unet
fi
git -C third_party/cpjku_unet checkout ismir-2020
export PYTHONPATH="$(pwd)/third_party/cpjku_unet:${PYTHONPATH:-}"

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt
CONFIG=/project/def-ichiro/pmohseni/music-alignment/configs/cpjku_native.yaml

echo "=== Step 1: Convert MSMD processed → CPJKU format (all splits) ==="
python -m mymodel.cpjku_adapter.convert \
    --processed "$PROC" \
    --out       "$CPJKU_FMT" \
    --splits    train val test

echo "=== Step 2: Create per-split symlink directories ==="
# train_model.py takes --train_set / --val_set as separate directories and
# globs all *.npz inside — it has no --split_file option.
python -m mymodel.cpjku_adapter.split_cpjku_data \
    --processed "$PROC" \
    --cpjku_fmt "$CPJKU_FMT"

TRAIN_DIR="$CPJKU_FMT/train"
VAL_DIR="$CPJKU_FMT/val"

echo "=== Step 3: Train CB_TA architecture from scratch (their train_model.py) ==="
# Architecture flags match CB_TA net_config.json exactly:
#   film_layers=[2..8], n_encoder_layers=4, n_filters_start=8,
#   rnn_size=128, spec_enc=32, rnn_layer=1, use_lstm=True, CBEncoder
#
# batch_size=1: our strips have variable widths; prepare_batch concatenates
#   along the batch axis and fails with mismatched widths if batch_size > 1.
#
# scale_factor=3: matches their original training setup (downscales strip
#   width and height by 3 before feeding to the network).
#
# dump_root: their code appends a timestamp, so the final model directory
#   will be results/cpjku_native/train_<timestamp>/best_model.pt
python third_party/cpjku_unet/audio_conditioned_unet/train_model.py \
    --train_set       "$TRAIN_DIR" \
    --val_set         "$VAL_DIR" \
    --config          "$CONFIG" \
    --film_layers     2 3 4 5 6 7 8 \
    --n_encoder_layers 4 \
    --n_filters_start 8 \
    --rnn_size        128 \
    --spec_enc        32 \
    --rnn_layer       1 \
    --use_lstm \
    --audio_encoder   CBEncoder \
    --scale_factor    3 \
    --batch_size      1 \
    --seq_len         16 \
    --learning_rate   1e-4 \
    --weight_decay    1e-5 \
    --patience        5 \
    --seed            4711 \
    --log_root        results/cpjku_native/runs \
    --dump_root       results/cpjku_native \
    --tag             msmd_strips \
    2>&1 | tee results/cpjku_native/train.log

echo "Training finished at $(date)"

echo "=== Step 4: Eval best trained model on test split ==="
# Find the most recently written best_model.pt under results/cpjku_native/
BEST=$(find results/cpjku_native -name 'best_model.pt' -newer results/cpjku_native/train.log 2>/dev/null | head -1)
if [ -z "$BEST" ]; then
    # Fallback: newest best_model.pt in the directory
    BEST=$(find results/cpjku_native -name 'best_model.pt' | sort | tail -1)
fi

if [ -z "$BEST" ]; then
    echo "ERROR: no best_model.pt found after training"; exit 1
fi
echo "Evaluating: $BEST"

python -m mymodel.cpjku_adapter.split_cpjku_data \
    --processed "$PROC" --cpjku_fmt "$CPJKU_FMT"   # idempotent

python third_party/cpjku_unet/audio_conditioned_unet/eval_model.py \
    --param_path   "$BEST" \
    --test_dir     "$CPJKU_FMT" \
    --config       "$CONFIG" \
    --split_file   "$CPJKU_FMT/split_test.yaml" \
    --scale_factor 3 \
    --batch_size   1 \
    --seq_len      128 \
    --eval_onsets \
    --piecewise_stats \
    2>&1 | tee results/cpjku_native/eval_trained_test.log

echo "Job finished at $(date)"
