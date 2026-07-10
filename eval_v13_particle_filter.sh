#!/bin/bash
#SBATCH --job-name=eval-v13-pf
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v13_pf-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v13_pf-%j.log

# E1: decode retrofit on v13's EXISTING trained checkpoint, no retraining.
# Compares original threshold+center-of-mass decode vs particle filter vs
# offline DTW-over-marginals, all from the SAME per-frame forward pass.
# D2 proved decode-only changes give large gains (5.1%->23.7% pct@0.5s) on a
# weaker model; this tests whether it transfers to the project's best model.

echo "Job started on $(hostname) at $(date)"
nvidia-smi
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

python -m mymodel.v13_mert_unet.eval_particle_filter \
    --checkpoint /scratch/pmohseni/results/v13_mert_linear/best_model.pt \
    --config     configs/v13_mert_linear.yaml \
    --split      test \
    --mert_emb_root data/MSMD/mert_emb \
    --pf_process_noise_std 3.0 \
    --pf_init_std 2.0 \
    --dtw_band_frac 0.05

echo "Job finished at $(date)"
