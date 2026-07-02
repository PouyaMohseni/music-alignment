#!/bin/bash
#SBATCH --job-name=music-cadp-m01
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cadp_m01-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cadp_m01-%j.log

# CADP M01: DINOv2 (score columns) + MERT (audio) + sim matrix + expected_distance
# Step 1: precompute DINOv2 column features (if not done)
# Step 2: train M01 (frozen projections, ~400K params)
# Step 3: eval pct@0.5s on test split

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1   # DINOv2 + MERT are fully cached; no internet on compute nodes

OUT=/scratch/pmohseni/results/cadp_m01
mkdir -p $OUT

D2_DIR=data/MSMD/dinov2_emb
mkdir -p $D2_DIR
N_DONE=$(find $D2_DIR -name "*.npy" -type f 2>/dev/null | wc -l)
echo "DINOv2 features already computed: $N_DONE"
if [ "$N_DONE" -lt 400 ]; then
    echo "Precomputing DINOv2 features..."
    python scripts/precompute_dinov2.py \
        --processed data/MSMD/processed \
        --out_dir   $D2_DIR \
        --batch_size 64
    echo "DINOv2 precompute done."
fi

echo ""
echo "Training CADP M01..."
python -m mymodel.cadp.m01_train \
    --config configs/cadp_m01.yaml \
    train.out_dir=$OUT

echo ""
echo "Running eval on test split..."
python -m mymodel.cadp.m01_eval \
    --checkpoint $OUT/best_model.pt \
    --config     configs/cadp_m01.yaml \
    --split      test \
    --out_dir    $OUT/eval

echo "Job finished at $(date)"
