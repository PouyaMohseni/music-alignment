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
# Trained from scratch here (not warm-started from A0) for a clean ablation
# number; CB_TA-Ext.md notes it "can fine-tune on top of a converged base"
# if a warm start is later wanted -- pass --param_path to do that.

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
    --tag B3_inr_subpixel

echo ""
echo "Training finished at $(date)"
