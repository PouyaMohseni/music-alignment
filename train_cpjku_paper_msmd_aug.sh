#!/bin/bash
#SBATCH --job-name=cpjku-aug-train
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_aug_train-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_aug_train-%j.log

# Train CB_TA on the full msmd_aug_v1-1_no-audio dataset (697 pieces, 7 tempos).
# 11-tempo version OOMs at 64GB (18711 entries). 7 tempos = 11907 entries, which
# is slightly less than the working Zenodo CB_TA training (1890 pages × 7 = 13230).
# Comparison against:
#   - CPJKU pretrained model (their internal training data)
#   - train_cpjku_paper_CB_TA.sh (Zenodo subset: 168 pieces, 7 tempos)
#
# Prerequisites:
#   1. run setup_cpjku_paper_login.sh (FluidSynth)
#   2. run sbatch convert_msmd_aug_pages.sh  (produces data/MSMD/msmd_aug_cpjku_pages/)
#
# IMPORTANT: train_model.py (CPJKU's vendored code) has no epoch/optimizer resume —
# every job restarts from epoch 0. Every past run (16 attempts, 2026-06-25 to 06-30)
# either crashed early, OOMed, or hit the 24h wall, and their checkpoints were saved
# under /project/... where they were later lost (project dir is quota-constrained;
# all 16 old params/ dirs now contain only net_config.json, no .pt files). One run
# (64327784) DID train cleanly to epoch 33 with improving val loss before timing out
# — the approach works, it just never got to keep its progress across resubmissions.
# Fixed here: (1) checkpoints go to /scratch (no quota risk), (2) auto warm-start
# from the latest latest_model.pt on resubmission via --param_path. Note this is a
# WEIGHTS-ONLY warm start, not a true resume — optimizer state, LR schedule, and the
# epoch/patience counters all restart, but training does not start from random init.

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
DATA=/scratch/pmohseni/music-alignment/msmd_aug_cpjku_pages
OUT=/scratch/pmohseni/results/cpjku_aug/CB_TA

mkdir -p "$OUT/runs" "$OUT/params"

if [ ! -d "$DATA/score" ]; then
    echo "ERROR: $DATA/score not found. Run sbatch convert_msmd_aug_pages.sh first." >&2
    exit 1
fi

NPAGES=$(ls "$DATA/score/"*.npz 2>/dev/null | wc -l)
echo "Training on $NPAGES score pages from msmd_aug_v1-1_no-audio"
echo "Tempos: 500 750 1000 1250 1500 1750 2000 (7 variants, 128GB)"
echo "Val:    Zenodo msmd_valid (28 pieces, tempo_1000 only)"
echo ""

# Copy 7-tempo config into submodule (survives git submodule update)
cp /project/def-ichiro/pmohseni/music-alignment/configs/msmd_aug_7tempo.yaml \
   "$REPO/audio_conditioned_unet/configs/msmd_aug_7tempo.yaml"

cd "$REPO/audio_conditioned_unet"

# Warm-start from the latest checkpoint if a previous run left one (weights only —
# train_model.py has no true resume, so epoch/optimizer/LR-schedule state restarts).
PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo "Warm-starting from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    echo "No previous checkpoint found, training from scratch"
fi

python train_model.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$DATA" \
    --val_set   /scratch/pmohseni/music-alignment/msmd_val_cpjku_pages \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug_7tempo.yaml \
    --audio_encoder CBEncoder \
    --tag CB_TA_aug \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
echo "Best model: $OUT/params/CB_TA_aug*/best_model.pt"
