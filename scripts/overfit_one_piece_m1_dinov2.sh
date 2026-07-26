#!/bin/bash
#SBATCH --job-name=m1-dinov2-overfit
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/m1_dinov2_overfit-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/m1_dinov2_overfit-%j.log

# Decisive test: is frozen DINOv2-base (+ trainable projection/context
# adapter) usable as M1's score tower? Same overfit-one-piece bar Phase 1
# validated for the from-scratch CNN tower (frame_acc 0.997) and the failed
# MuSViT attempt (0.65, see M1.md). Unlike MuSViT, DINOv2's tiling scheme is
# already proven in this project (dinov2-full-encoder / mert-dinov2-crossattn
# jobs, currently training). Submitted as a real job (never run this kind of
# multi-tile-forward-pass test on the login node -- see
# memory/cluster_workflow.md, 2026-07-22 incident).

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m scripts.overfit_one_piece_m1_tower \
    --score_tower dinov2 \
    --config configs/d1_align_matrix.yaml \
    --steps 150 --limit 12 --t_max 300

echo "Job finished at $(date)"
