#!/bin/bash
#SBATCH --job-name=convert-val
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/convert_val-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/convert_val-%j.log

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate

python mymodel/cpjku_adapter/convert_msmd_aug_pages.py \
    --processed_root data/MSMD/processed \
    --aug_root       data/MSMD/msmd_aug_v1-1_no-audio \
    --output_root    /scratch/pmohseni/music-alignment/msmd_val_cpjku_pages \
    --workers        4 \
    --split_file     data/MSMD/processed/splits.json \
    --split_key      val

echo "Done at $(date)"
