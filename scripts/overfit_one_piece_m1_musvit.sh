#!/bin/bash
#SBATCH --job-name=m1-musvit-overfit
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/m1_musvit_overfit-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/m1_musvit_overfit-%j.log

# Decisive test: is frozen MuSViT (+ trainable projection/context adapter)
# usable as M1's score tower? Same overfit-one-piece bar Phase 1 already
# validated for the from-scratch CNN tower (frame_acc 0.997). Previously run
# directly on the login node (narval1) and killed mid-run after being flagged
# by Alliance Canada admins -- see memory/cluster_workflow.md, 2026-07-22
# incident. Never run this kind of multi-step forward-pass-through-a-large-
# transformer test outside a scheduled job again. GPU makes MuSViT's 14
# tile-forward-passes/step far cheaper than the ~52s/step seen on CPU.

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/overfit_one_piece_m1_musvit.py \
    --config configs/d1_align_matrix.yaml \
    --steps 120 --limit 12 --t_max 300

echo "Job finished at $(date)"
