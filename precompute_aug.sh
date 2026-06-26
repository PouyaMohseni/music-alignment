#!/bin/bash
#SBATCH --job-name=precompute-aug
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/precompute_aug-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/precompute_aug-%j.log

# Precompute multi-tempo MERT embeddings for v3_aug retraining.
# For each of 467 processed/ pieces, synthesizes audio at 11 tempos with
# FluidSynth (grand-piano-YDP, same soundfont as CPJKU paper) then runs frozen
# MERT-v1-95M. Tile (ViT) embeddings are reused from the existing
# data/MSMD/embeddings/ cache — only audio changes across tempos.
#
# Output: data/MSMD/embeddings_aug/  (~467 × 11 = 5137 .npz files)
# Estimated time: 8-12 hours on A100.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

# FluidSynth
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}

# Prevent BLAS fork deadlock
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# HuggingFace model cache (MERT + ViT weights, already downloaded)
export TRANSFORMERS_CACHE=/scratch/pmohseni/hf_cache
export HF_HOME=/scratch/pmohseni/hf_cache

mkdir -p results data/MSMD/embeddings_aug

python -m mymodel.v3_fullseq.precompute_aug \
    --processed  data/MSMD/processed \
    --emb_cache  data/MSMD/embeddings \
    --out        data/MSMD/embeddings_aug \
    --sf         third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2 \
    --config     configs/v3_aug.yaml \
    --chunk_sec  5.0

echo "Job finished at $(date)"
