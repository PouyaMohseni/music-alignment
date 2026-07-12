#!/bin/bash
#SBATCH --job-name=f3-pf
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f3_pf-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/f3_pf-%j.log

# F3 variant: particle-filter decode on the ensemble-averaged marginals
# (v13+v14+v15), alongside original threshold+CoM in the same pass. E1 found
# particle filter HURT a converged single model (v13: 66.1%->57.6%); testing
# whether pre-averaging across 3 models changes that.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --fusion mean --decoders original,particle_filter \
    --split test --out_dir results/f3_ensemble_decode/pf_v13+v14+v15

echo "Job finished at $(date)"
