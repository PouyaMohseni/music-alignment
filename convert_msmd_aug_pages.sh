#!/bin/bash
#SBATCH --job-name=convert-pages
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/convert_pages-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/convert_pages-%j.log

set -euo pipefail
echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate

python mymodel/cpjku_adapter/convert_msmd_aug_pages.py \
    --processed_root data/MSMD/processed \
    --aug_root       data/MSMD/msmd_aug_v1-1_no-audio \
    --output_root    data/MSMD/msmd_aug_cpjku_pages \
    --workers        8

echo "Done at $(date)"
