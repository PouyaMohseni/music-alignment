#!/bin/bash
#SBATCH --job-name=analyze-only
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/analyze_only-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/analyze_only-%j.log

# Quick check of partial per-onset-diag-full results. Submitted as a proper
# batch job rather than run interactively -- the login node is currently
# under enough contention that even `import torch` was hanging there.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python scripts/per_onset_error_diagnostic.py --analyze_only --out results/per_onset_diagnostic_full.json

echo "Job finished at $(date)"
