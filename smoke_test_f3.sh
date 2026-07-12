#!/bin/bash
#SBATCH --job-name=smoke-f3
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f3-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f3-%j.log

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

echo "=== smoke: 3-model + both decoders, 5 pieces ==="
python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15 --decoders original,offline_dtw --split test --limit 5 --out_dir results/f3_smoke/3model

echo "=== smoke: 4-model (+v13_midi, down-weighted) + both decoders, 5 pieces ==="
python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15,v13_midi --weights 0.3,0.3,0.3,0.1 --decoders original,offline_dtw --split test --limit 5 --out_dir results/f3_smoke/4model

echo "Job finished at $(date)"
