#!/bin/bash
#SBATCH --job-name=b6-impulse-response
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/b6_impulse_response-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/b6_impulse_response-%j.log

# CB_TA-Ext B6: synthetic impulse-response + pink-noise augmentation applied
# to the synthesized training audio, on the same data/config as A0. Uses the
# ORIGINAL CBEncoder (no other change) so this ablation isolates the data
# augmentation alone. No auxiliary loss (unlike B2-B5) -- see
# extensions/hooks/run_train_b6.py / extensions/augmentation/impulse_response.py
# for why synthetic IRs are used instead of a real IR database (no internet
# access on compute nodes).

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
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B6_impulse_response
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


export B6_AUG_PROB=0.5
export B6_SNR_LO=10.0
export B6_SNR_HI=30.0
export B6_N_IRS=16

echo "=== B6: synthetic impulse-response augmentation (same data/config as A0) ==="
echo "aug_prob=$B6_AUG_PROB snr_range=[$B6_SNR_LO,$B6_SNR_HI]dB n_irs=$B6_N_IRS"
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_b6.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set /scratch/pmohseni/msmd_train_full \
    --val_set   ../data/msmd/msmd_valid \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder CBEncoder \
    --tag B6_impulse_response \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
echo "Best model: $OUT/params/*_B6_impulse_response/best_model.pt"
