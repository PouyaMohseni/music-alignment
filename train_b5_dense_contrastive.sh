#!/bin/bash
#SBATCH --job-name=b5-contrastive
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/b5_contrastive-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/b5_contrastive-%j.log

# CB_TA-Ext B5: dense contrastive auxiliary loss at decoder stage 7 (see
# run_train_b5.py for why not the spec's literal "decoder_5" -- that stage
# doesn't exist under this network's actual FiLM numbering), on the same
# data/config as A0. Uses the ORIGINAL CBEncoder, isolating this ablation.

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
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B5_dense_contrastive_aux
mkdir -p "$OUT/runs" "$OUT/params"

echo "=== B5: dense contrastive auxiliary loss (same data/config as A0) ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_b5.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set /scratch/pmohseni/msmd_train_full \
    --val_set   ../data/msmd/msmd_valid \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder CBEncoder \
    --tag B5_dense_contrastive_aux

echo ""
echo "Training finished at $(date)"
