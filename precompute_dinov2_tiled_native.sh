#!/bin/bash
#SBATCH --job-name=precompute-dinov2-tiled
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/precompute_dinov2_tiled-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/precompute_dinov2_tiled-%j.log

# Resolution-preserving DINOv2 precompute for the full-encoder-replacement
# experiment: tiles each native page at 224x224 (no whole-page downscaling
# first), unlike precompute_dinov2_native_pages.py's whole-page-squash
# version which would blur fine sheet-music detail (staff lines, noteheads)
# across each of only 16x16 patches. ~24 tiles/page average (verified on
# real pages: 1181x835 -> 6x4=24 tiles), 1098 pages total -- ~26k DINOv2
# forward passes, more expensive than the whole-page version but still
# modest.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

python scripts/precompute_dinov2_tiled_native.py \
    --score_dirs /scratch/pmohseni/msmd_train_full/score \
                 third_party/cpjku_unet/data/msmd/msmd_valid/score \
                 third_party/cpjku_unet/data/msmd/msmd_test/score \
    --out_dir /scratch/pmohseni/dinov2_emb_tiled_native \
    --tile_size 224 --stride 224

echo "Job finished at $(date)"
