#!/bin/bash
#SBATCH --job-name=cpjku-aug-train
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
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
#   2. run sbatch convert_msmd_aug.sh  (produces data/MSMD/msmd_aug_cpjku/)

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

git submodule update --init third_party/cpjku_unet || true
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
DATA=/scratch/pmohseni/music-alignment/msmd_aug_cpjku
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_aug/CB_TA

mkdir -p "$OUT/runs" "$OUT/params"

if [ ! -d "$DATA/score" ]; then
    echo "ERROR: $DATA/score not found. Run sbatch convert_msmd_aug.sh first." >&2
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

python train_model.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$DATA" \
    --val_set   ../data/msmd/msmd_valid \
    --use_lstm \
    --augment \
    --batch_size 1 \
    --config    configs/msmd_aug_7tempo.yaml \
    --audio_encoder CBEncoder \
    --tag CB_TA_aug

echo ""
echo "Training finished at $(date)"
echo "Best model: $OUT/params/CB_TA_aug*/best_model.pt"
