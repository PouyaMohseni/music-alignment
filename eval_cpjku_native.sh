#!/bin/bash
#SBATCH --job-name=cpjku-native-eval
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_native_eval-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_native_eval-%j.log

# Run their unmodified eval_model.py with the CB_TA pretrained model on our
# MSMD test split.  No adapter, no patches — their exact pipeline.
#
# Prerequisites:
#   sbatch setup_cpjku310.sh          # create .venv_cpjku310
#   sbatch this script                 # after setup job finishes

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/cpjku_native

module load gcc opencv python/3.10
source /scratch/pmohseni/venv_cpjku310/bin/activate

# Prevent BLAS-fork deadlock: their eval_model.py uses multiprocessing.Pool(8)
# to load data; forked workers inherit locked BLAS thread pools and hang.
# Setting threads=1 makes BLAS single-threaded → no pool → no deadlock.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Ensure submodule is on the right branch
git submodule update --init third_party/cpjku_unet
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..
export PYTHONPATH="$(pwd)/third_party/cpjku_unet:${PYTHONPATH:-}"

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt
CONFIG=/project/def-ichiro/pmohseni/music-alignment/configs/cpjku_native.yaml
MODEL=third_party/cpjku_unet/models/CB_TA/best_model.pt

echo "=== Step 1: Convert MSMD processed → CPJKU format (all splits) ==="
python -m mymodel.cpjku_adapter.convert \
    --processed "$PROC" \
    --out       "$CPJKU_FMT" \
    --splits    train val test

echo "=== Step 2: Build split YAML for test ==="
python -c "
import json, yaml
from pathlib import Path
proc = Path('$PROC')
out  = Path('$CPJKU_FMT')
splits = json.load(open(proc / 'splits.json'))
available = [p for p in splits['test'] if (out / 'score' / (p + '.npz')).exists()]
print(f'test split: {len(available)} pieces')
yaml.dump({'files': available}, open(out / 'split_test.yaml', 'w'))
"

echo "=== Step 3: Run CB_TA eval (their unmodified eval_model.py) ==="
# --eval_onsets: report % of onset frames within time thresholds (their main metric)
# --scale_factor 3: matches their CB_TA training setup
# --batch_size 1: our strips have variable widths; batching > 1 fails concat
# --seq_len 128: chunk size for BPTT (memory vs speed trade-off)
python third_party/cpjku_unet/audio_conditioned_unet/eval_model.py \
    --param_path   "$MODEL" \
    --test_dir     "$CPJKU_FMT" \
    --config       "$CONFIG" \
    --split_file   "$CPJKU_FMT/split_test.yaml" \
    --scale_factor 3 \
    --batch_size   1 \
    --seq_len      128 \
    --eval_onsets \
    --piecewise_stats \
    2>&1 | tee results/cpjku_native/eval_CB_TA_test.log

echo "Job finished at $(date)"
