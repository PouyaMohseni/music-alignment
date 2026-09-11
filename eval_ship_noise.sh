#!/bin/bash
#SBATCH --job-name=shipnoise
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/shipnoise-%j.log
# Held-out gain of nbrp64_s0 over the featureless selector at four noise
# levels, all on the FEATK=256 supply. Submit after dump_hv_noisy256_more.sh.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python extensions/analysis/heldout_compare.py --dir /scratch/pmohseni/omr/candhv256 \
    --cand /scratch/pmohseni/omr/scorer/nbr/nbrp64_s0.pt --snrs 12,6,3,0.5
