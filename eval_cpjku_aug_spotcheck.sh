#!/bin/bash
#SBATCH --job-name=eval-cpjku-aug-spot
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_cpjku_aug_spot-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_cpjku_aug_spot-%j.log

# Spot-check eval of the in-progress cpjku-aug-train checkpoint (only a few
# epochs in as of submission) — just to confirm the eval pipeline works
# against it and get an early read, not a final result.

echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
    git submodule update --init third_party/cpjku_unet
fi
git -C third_party/cpjku_unet checkout ismir-2020

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt

CKPT_DIR=$(find /scratch/pmohseni/results/cpjku_aug/CB_TA/params -maxdepth 1 -name "*_CB_TA_aug" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"

python -m mymodel.cpjku_adapter.eval_official \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --param_path  "$CKPT_DIR/best_model.pt" \
    --net_config  "$CKPT_DIR/net_config.json" \
    --split       test \
    --batch_size  1 \
    --seq_len     8 \
    --scale_factor 3

echo "Job finished at $(date)"
