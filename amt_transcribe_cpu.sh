#!/bin/bash
#SBATCH --job-name=amt-transcribe-cpu
#SBATCH --account=def-ichiro

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/amt_transcribe_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/amt_transcribe_cpu-%j.log

# Transcribe the 25 real-performance pages (room + di-left) with the stock Kong
# AMT model and the reverb-augmented Edwards model.  ~25 min of audio per tier,
# 4 (model x tier) passes = ~100 min of audio total; trivial on a GPU.
# See scripts/amt_transcribe_real.py for the why.

echo "Job started on $(hostname) at $(date)"
nvidia-smi || echo "(no gpu)"

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export OMP_NUM_THREADS=8
export MPLBACKEND=Agg

python scripts/amt_transcribe_real.py --device cpu

echo "Job finished at $(date)"
