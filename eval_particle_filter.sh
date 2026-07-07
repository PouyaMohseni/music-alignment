#!/bin/bash
#SBATCH --job-name=eval-particle-filter
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_particle_filter-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_particle_filter-%j.log

# C3: Bayesian particle-filter decoder on top of CB_TA's own bundled
# pretrained model (no retraining -- decode-only change). Compare this run's
# pct@0.5s directly against eval_cpjku_official.sh's result on the SAME
# checkpoint to isolate the decoder's contribution.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

if [ ! -f third_party/cpjku_unet/network.py ]; then
    git submodule update --init third_party/cpjku_unet
fi
git -C third_party/cpjku_unet checkout ismir-2020

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt

python -m mymodel.cpjku_adapter.eval_particle_filter \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --model       CB_TA \
    --split       test \
    --seq_len     8 \
    --scale_factor 3

echo "Job finished at $(date)"
