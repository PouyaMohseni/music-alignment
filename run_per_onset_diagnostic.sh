#!/bin/bash
#SBATCH --job-name=per-onset-diag
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/per_onset_diag-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/per_onset_diag-%j.log

# Per-ONSET (not per-piece) error diagnostic on our best model (F4) across
# the worst/shared pieces -- tests the repeat-ambiguity and sparse-audio
# failure hypotheses directly against per-onset error, and finds WHERE in
# each piece (relative position) the worst errors cluster.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python scripts/per_onset_error_diagnostic.py

echo "Job finished at $(date)"
