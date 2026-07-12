#!/bin/bash
#SBATCH --job-name=f5-conf-vel
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f5_conf_vel-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/f5_conf_vel-%j.log

# F5: confidence-gated velocity-prior decode, targeting the two cross-model-
# validated failure clusters (repeat ambiguity + sparse audio). One forward
# pass, whole cv hyperparameter grid decoded from the same marginals, plus
# original + hybrid_snap(0.2) baselines for direct comparison.
#   cv_w<window>_g<gate>: window = +-cols for peak concentration, gate = conf floor.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 \
    --decoders original,hybrid_snap,cv_w3_g0.3,cv_w5_g0.3,cv_w10_g0.3,cv_w20_g0.3,cv_w5_g0.5,cv_w10_g0.5 \
    --snap_frac 0.2 \
    --split test \
    --out_dir results/f3_ensemble_decode/f5_confidence_velocity

echo "Job finished at $(date)"
