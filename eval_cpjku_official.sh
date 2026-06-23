#!/bin/bash
#SBATCH --job-name=cpjku-eval
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official-%j.log

echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate

# Make sure submodule is checked out
git submodule update --init third_party/cpjku_unet
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt
mkdir -p results

echo "=== Step 1: Convert MSMD processed → CPJKU format ==="
python -m mymodel.cpjku_adapter.convert \
    --processed $PROC \
    --out       $CPJKU_FMT \
    --splits    test val train

echo "=== Step 2: Eval with CPJKU pre-trained CB_TA model ==="
python -m mymodel.cpjku_adapter.eval_official \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --model       CB_TA \
    --split       test \
    --batch_size  4 \
    --scale_factor 1

echo "Job finished at $(date)"
