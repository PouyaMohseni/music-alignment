#!/bin/bash
#SBATCH --job-name=msmd-all-perf
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_fullseq/allperf-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_fullseq/allperf-%j.log

# Stage 1: build the all-performances processed dataset (strips shared via
# symlink, audio synthesised per tempo/soundfont variant). Output to SCRATCH
# (thousands of small files — fine on scratch's 1M inode quota, NOT /project).
echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v3_fullseq

source .venv/bin/activate
export PATH=~/miniforge/envs/msmd-tools/bin:$PATH   # fluidsynth

python -m msmd_prep.run_all \
  --raw    data/MSMD/msmd_aug_v1-1_no-audio \
  --splits data/MSMD/msmd/msmd/splits/all_split.yaml \
  --out    /lustre07/scratch/pmohseni/music-alignment/data/MSMD/processed_all \
  --sf2    ~/sf2/MuseScore_General.sf3 \
  --all-performances \
  --jobs   32

echo "Job finished at $(date)"
