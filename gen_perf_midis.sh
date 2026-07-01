#!/bin/bash
#SBATCH --job-name=gen-perf-midis
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/gen_perf_midis-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/gen_perf_midis-%j.log

# Generate missing tempo-augmented performance MIDIs for all 945 msmd_train pages.
# Prereq: mido is installed in venv_cpjku310 (it is — used by madmom as a dep).
# CPU-only, no GPU needed. Takes ~5-10 min.
# After this completes, resubmit train_cpjku_paper_CB_TA.sh.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"

source /scratch/pmohseni/venv_cpjku310/bin/activate

cd /project/def-ichiro/pmohseni/music-alignment

python generate_msmd_train_perf_midis.py

echo "Done at $(date)"
