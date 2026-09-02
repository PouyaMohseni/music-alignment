#!/bin/bash
#SBATCH --job-name=proxy
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/proxy-%j.log
# Which set we can select on, now that synthetic validation cannot be trusted.
#
# Detector argmax, i.e. difficulty independent of any decoder:
#   ROOM 80.0 | do 83.6 | rp_synth 87.2 | synth valid 90.7 | held-out 80pc 97.3
#
# Over seven variants, synthetic-validation rank is ANTI-correlated with room
# (Spearman -0.39): it picks vel_feat, worth 90.0, over feat_wide, worth 94.0.
# `do` is real audio 3.6 points from room. It shares the sixteen pieces, so
# piece identity leaks and it is NOT a clean holdout -- but the acoustic
# condition genuinely differs, which is the axis synthetic validation fails on.
# If ranking on `do` predicts ranking on room, it is usable for selection with
# that caveat stated. If it does not, no set we own is usable and the noisy
# dump is the only way forward.
#
# The rollout must reproduce the harness first: ir_only at blend 0.7 on room is
# 91.4 and at blend 0.0 is 86.5, or nothing below counts.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
# tiny per-frame tensors: threads are pure overhead here
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u extensions/analysis/proxy_rank.py
echo ""; echo "Job finished at $(date)"
