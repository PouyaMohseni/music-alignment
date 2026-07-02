#!/bin/bash
#SBATCH --job-name=eval-cadp-m01
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_cadp_m01-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_cadp_m01-%j.log

set -euo pipefail
echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

OUT=/scratch/pmohseni/results/cadp_m01
D2_DIR=/scratch/pmohseni/dinov2_emb

python -m mymodel.cadp.m01_eval \
    --checkpoint $OUT/best_model.pt \
    --config     configs/cadp_m01.yaml \
    --split      test \
    --out_dir    $OUT/eval \
    data.dinov2_root=$D2_DIR

echo "Job finished at $(date)"
