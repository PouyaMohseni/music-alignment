#!/bin/bash
#SBATCH --job-name=f3-fusion-max
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f3_fusion_max-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/f3_fusion_max-%j.log

# F3 variant: max fusion (confident-vote) across v13+v14+v15 -- if any
# member is confidently right on a frame, trust it rather than averaging it
# down against two more uncertain members.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --fusion max --decoders original \
    --split test --out_dir results/f3_ensemble_decode/max_v13+v14+v15

echo "Job finished at $(date)"
