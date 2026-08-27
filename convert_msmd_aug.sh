#!/bin/bash
#SBATCH --job-name=msmd-convert
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/msmd_convert-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/msmd_convert-%j.log

# Convert msmd_aug_v1-1_no-audio → CPJKU flat format.
# Runs fast (~5 min) — CPU only, no GPU needed.
# After this finishes, submit train_cpjku_paper_msmd_aug.sh.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
source /scratch/pmohseni/venv_cpjku310/bin/activate

OUTDIR=/scratch/pmohseni/music-alignment/msmd_aug_cpjku

python3 convert_msmd_aug_to_cpjku.py \
    --src_dir data/MSMD/msmd_aug_v1-1_no-audio \
    --out_dir "$OUTDIR" \
    --workers 16

echo ""
echo "Conversion finished at $(date)"
ls "$OUTDIR/score/" | wc -l
wc -l < "$OUTDIR/split_all.yaml"
