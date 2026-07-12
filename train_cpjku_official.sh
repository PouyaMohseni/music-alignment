#!/bin/bash
#SBATCH --job-name=cpjku-train
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_train-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_train-%j.log

# Train the CPJKU ConditionalUNet (Henkel et al. ISMIR 2020) on our MSMD strips
# using THEIR exact training machinery (iterate_dataset, dice_loss, scheduler),
# then evaluate the best checkpoint on the test split with THEIR exact metric.

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

# Alliance cluster: opencv must come from the module system, not pip.
module load gcc opencv
source .venv/bin/activate

# Submodule (their repo) + correct branch.
if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
    git submodule update --init third_party/cpjku_unet
fi
git -C third_party/cpjku_unet checkout ismir-2020

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt
DUMP=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official
mkdir -p results "$DUMP"

echo "=== Step 1: Convert MSMD processed -> CPJKU format (train/val/test) ==="
python -m mymodel.cpjku_adapter.convert \
    --processed $PROC \
    --out       $CPJKU_FMT \
    --splits    train val test

echo "=== Step 2: Train CPJKU ConditionalUNet (CB_TA config) on our strips ==="
python -m mymodel.cpjku_adapter.train_official \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --dump_root   $DUMP \
    --tag         CB_TA_strips \
    --seq_len     16 \
    --scale_factor 3 \
    --augment

echo "Training finished at $(date). Locating best checkpoint..."

# Most recent run directory under the dump root.
RUN_DIR=$(ls -dt $DUMP/*/ 2>/dev/null | head -1)
CKPT=${RUN_DIR}best_model.pt
NETCFG=${RUN_DIR}net_config.json
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no best_model.pt found in $RUN_DIR"; exit 1
fi
echo "Evaluating: $CKPT"

echo "=== Step 3: Eval trained model on test split (their exact metric) ==="
python -m mymodel.cpjku_adapter.eval_official \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --param_path  "$CKPT" \
    --net_config  "$NETCFG" \
    --split       test \
    --seq_len     8 \
    --scale_factor 3

echo "Job finished at $(date)"
