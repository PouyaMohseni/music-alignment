#!/bin/bash
#SBATCH --job-name=idxprior
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/idxprior-%j.log
# Decode with the transition prior in NOTE-INDEX space instead of pixels.
#
# Measured on room, the true step from one onset to the next has CV 0.425 in
# pixels and 0.256 in note index -- 40% more predictable, in 14 of 16 pieces --
# because engraving packs a run of sixteenths into the width of one whole note,
# so a fixed sigma_px is too tight in sparse passages and too loose in dense
# ones. Drift errors sit at a median 97.6 px, the wrong NOTEHEAD in the same or
# an adjacent bar, which is what a pixel prior discriminates worst.
#
# The index comes from the detector's own candidates clustered into bins; no
# annotation, no pitches. GATE: pixel mode must reproduce 86.5.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python -u extensions/analysis/run_index_sweep.py
echo ""; echo "Job finished at $(date)"
