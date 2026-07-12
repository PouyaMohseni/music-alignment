#!/bin/bash
#SBATCH --job-name=eval-f3-3model
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_f3_3model-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_f3_3model-%j.log

# F3 variant A: same 3 members as F2 (v13+v14+v15), but decode BOTH ways
# (original threshold+CoM, which reproduces F2's 72.0% exactly, AND
# offline-DTW on the pre-averaged marginals) in one pass, to see whether DTW
# decode helps once heatmap noise is already reduced by ensembling -- E1
# found DTW hurt pct@0.5s for a single converged model (v13: 66.1%->55.4%)
# but improved mean error; testing if that trade still holds post-ensemble.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 \
    --decoders original,offline_dtw \
    --split test \
    --out_dir results/f3_ensemble_decode/v13+v14+v15

echo "Job finished at $(date)"
