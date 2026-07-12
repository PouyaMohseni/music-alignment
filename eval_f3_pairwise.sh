#!/bin/bash
#SBATCH --job-name=f3-pairwise
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f3_pairwise-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/f3_pairwise-%j.log

# F3 diagnostic: pairwise 2-model ensembles (v13+v14, v13+v15, v14+v15) --
# which pair carries F2's gain, and does dropping to 2 members still beat
# the best single model (v15 at 66.4%)?

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

echo "=== v13+v14 ==="
python -m mymodel.f3_ensemble_decode.eval --models v13,v14 --decoders original --split test --out_dir results/f3_ensemble_decode/v13+v14

echo "=== v13+v15 ==="
python -m mymodel.f3_ensemble_decode.eval --models v13,v15 --decoders original --split test --out_dir results/f3_ensemble_decode/v13+v15

echo "=== v14+v15 ==="
python -m mymodel.f3_ensemble_decode.eval --models v14,v15 --decoders original --split test --out_dir results/f3_ensemble_decode/v14+v15

echo "Job finished at $(date)"
