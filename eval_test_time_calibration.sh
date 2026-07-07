#!/bin/bash
#SBATCH --job-name=eval-c5-calib
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_c5_calib-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_c5_calib-%j.log

# C5: test-time per-piece calibration, applied on top of CPJKU's own bundled
# pretrained CB_TA model (no retraining) -- isolates this decode-time
# technique's contribution, directly comparable to eval_cpjku_official.sh's
# result on the SAME checkpoint.

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
mkdir -p results

echo "=== C5: test-time per-piece calibration on CPJKU pretrained CB_TA ==="
python -m mymodel.cpjku_adapter.eval_test_time_calibration \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --model       CB_TA \
    --split       test \
    --seq_len     8 \
    --scale_factor 3 \
    --calib_seconds 8.0 \
    --calib_steps   15 \
    --calib_lr      1e-3

echo "Job finished at $(date)"
