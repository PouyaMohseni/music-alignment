#!/bin/bash
#SBATCH --job-name=eval-b3
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_b3-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_b3-%j.log

# Interim spot-check of B3_inr_subpixel on its current best_model.pt -- training is
# still in progress as of submission, so this is not a final result. Uses
# the generic eval_official.py path since B3_inr_subpixel's checkpoint is a plain
# CBEncoder-based ConditionalUNet (aux heads/losses only affected training,
# not the base network's segmentation forward pass).

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt

CKPT_DIR=$(find results/cb_ta_ext/B3_inr_subpixel/params -maxdepth 1 -name "*_B3_inr_subpixel" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"

python -m mymodel.cpjku_adapter.eval_official \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --param_path  "$CKPT_DIR/best_model.pt" \
    --net_config  "$CKPT_DIR/net_config.json" \
    --split       test \
    --scale_factor 3

echo "Job finished at $(date)"
