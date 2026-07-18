#!/bin/bash
#SBATCH --job-name=precompute-dinov2-native
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/precompute_dinov2_native-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/precompute_dinov2_native-%j.log

# Precomputes DINOv2 page embeddings (CLS + 16x16 patch grid) for all
# native MSMD score pages (train_full/valid/test, 1098 pages total). Runs
# in the main .venv (has torch+transformers); venv_cpjku310 (used for
# actual training) lacks transformers, confirmed directly, so this can't
# run live during training -- same constraint MERT precompute already had.
# Much cheaper than the MERT audio precompute: no FluidSynth rendering, one
# embedding per page (not per tempo), ~1098 forward passes total.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

python scripts/precompute_dinov2_native_pages.py \
    --score_dirs /scratch/pmohseni/msmd_train_full/score \
                 third_party/cpjku_unet/data/msmd/msmd_valid/score \
                 third_party/cpjku_unet/data/msmd/msmd_test/score \
    --out_dir /scratch/pmohseni/dinov2_emb_native

echo "Job finished at $(date)"
