#!/bin/bash
#SBATCH --job-name=music-cadp-m04
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cadp_m04-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cadp_m04-%j.log

# CADP M04: dense tokens (no audio pooling) + DTW decode. Standalone ablation
# entry to isolate "dense tokens" from M05's "dense tokens + learned decode".
# DTW eval on dense (unchunked) matrices can take a while for the widest
# pieces (pure-Python backtrack) -- 24h wall covers train + eval comfortably.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

OUT=/scratch/pmohseni/results/cadp_m04
mkdir -p $OUT

D2_DIR=/scratch/pmohseni/dinov2_emb
mkdir -p $D2_DIR
N_DONE=$(find $D2_DIR -name "*.npy" -type f 2>/dev/null | wc -l)
echo "DINOv2 features already computed: $N_DONE"
if [ "$N_DONE" -lt 467 ]; then
    echo "Precomputing DINOv2 features..."
    python scripts/precompute_dinov2.py \
        --processed data/MSMD/processed \
        --out_dir   $D2_DIR \
        --batch_size 64
    echo "DINOv2 precompute done."
fi

echo ""
echo "Training CADP M04..."
python -m mymodel.cadp.m04_train \
    --config configs/cadp_m04.yaml \
    train.out_dir=$OUT \
    data.dinov2_root=$D2_DIR

echo ""
echo "Running eval on test split..."
python -m mymodel.cadp.m04_eval \
    --checkpoint $OUT/best_model.pt \
    --config     configs/cadp_m04.yaml \
    --split      test \
    --out_dir    $OUT/eval \
    data.dinov2_root=$D2_DIR

echo "Job finished at $(date)"
