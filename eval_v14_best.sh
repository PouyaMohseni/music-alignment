#!/bin/bash
#SBATCH --job-name=eval-v14-best
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v14_best-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v14_best-%j.log
set -euo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1
python -m mymodel.v13_mert_unet.eval \
    --checkpoint /scratch/pmohseni/results/v14_mert_bilstm/best_model.pt \
    --config     configs/v14_mert_bilstm.yaml \
    --split      test \
    --out_dir    /scratch/pmohseni/results/v14_mert_bilstm/eval_best
