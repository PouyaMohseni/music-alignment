#!/bin/bash
#SBATCH --job-name=per-onset-diag-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/per_onset_diag_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/per_onset_diag_cpu-%j.log

# CPU-only variant of run_per_onset_diagnostic.sh -- the GPU version
# (job 65371156) has been stuck PENDING for hours on the heavily
# oversubscribed A100 partition. This diagnostic doesn't actually need a
# GPU: it's a small 3-model ensemble (each a modest ConditionalUNet) run
# over already-precomputed MERT audio embeddings, on only 14 pieces --
# perfectly workable on CPU, just slower per forward pass. Routes around
# the GPU queue entirely via the much-less-contended CPU partition.
# The script already auto-falls back to CPU when torch.cuda.is_available()
# is False, so no code changes needed.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

python scripts/per_onset_error_diagnostic.py

echo "Job finished at $(date)"
