#!/bin/bash
#SBATCH --job-name=eval-v2-dtw-now-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v2_dtw_now_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v2_dtw_now_cpu-%j.log

# CPU variant -- GPU queue is severely backlogged (588 running / 3529 pending)
# vs CPU partitions (2883 running / 631 pending). v2_crossattn uses LIVE
# MERT+ViT encoders (not precomputed), so this is slower per-piece than a
# precomputed-embedding eval, hence the generous 6h budget -- but the CPU
# queue itself should start almost immediately, which likely beats waiting
# out the GPU backlog in real wall-clock terms.

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "=== Eval v2_crossattn_dtw (real DTW-phase training, just completed) -- CPU ==="
python -m mymodel.v2_crossattn.eval \
    --checkpoint results/v2_crossattn_dtw/checkpoint_010000.pt \
    --config configs/v2_crossattn_dtw.yaml \
    --split test

echo "Job finished at $(date)"
