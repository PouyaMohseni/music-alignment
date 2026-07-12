#!/bin/bash
#SBATCH --job-name=smoke-f3-v2
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f3_v2-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f3_v2-%j.log

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

echo "=== smoke: median fusion, 3 pieces ==="
python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15 --fusion median --decoders original --split test --limit 3 --out_dir results/f3_smoke/median

echo "=== smoke: max fusion, 3 pieces ==="
python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15 --fusion max --decoders original --split test --limit 3 --out_dir results/f3_smoke/max

echo "=== smoke: particle_filter decoder, 3 pieces ==="
python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15 --decoders original,particle_filter --split test --limit 3 --out_dir results/f3_smoke/pf

echo "=== smoke: pairwise v13+v14, 3 pieces ==="
python -m mymodel.f3_ensemble_decode.eval --models v13,v14 --decoders original --split test --limit 3 --out_dir results/f3_smoke/pairwise

echo "Job finished at $(date)"
