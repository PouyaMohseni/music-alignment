#!/bin/bash
#SBATCH --job-name=eval-b1a
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_b1a-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_b1a-%j.log

# Interim spot-check of B1a (MERT audio-encoder swap) on its current
# best_model.pt -- training is still in progress (only a few epochs in as
# of submission), so this is not a final result.

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

CKPT_DIR=$(find results/cb_ta_ext/B1a_mert_swap/params -maxdepth 1 -name "*_B1a_mert_swap" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"

python -m mymodel.cpjku_adapter.eval_official_mert \
    --mert_root   /scratch/pmohseni/mert_emb_zenodo/cpjku_fmt_test_eval \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --param_path  "$CKPT_DIR/best_model.pt" \
    --net_config  "$CKPT_DIR/net_config.json" \
    --split       test \
    --scale_factor 3

echo "Job finished at $(date)"
