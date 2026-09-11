#!/bin/bash
#SBATCH --job-name=shipoff
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/shipoff-%j.log
# Every table on the two pages that was computed for vel_p8, recomputed for
# nbrp64_s0: parameter count, per-piece ladder, the error decomposition, and
# the held-out seed table on the FEATK=256 supply the selection was scored on.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
M=/scratch/pmohseni/omr/scorer
S=$M/nbr/nbrp64_s0.pt
echo "##### parameters"
python - "$S" <<'PY'
import sys
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.heads.cand_scorer import load
for p in (sys.argv[1], '/scratch/pmohseni/omr/scorer/grid/vel_p8.pt'):
    m = load(p)[0]
    print(p.split('/')[-1], sum(t.numel() for t in m.parameters()),
          'nf', m.nf, 'featdim', m.featdim)
PY
echo ""; echo "##### per-piece"
SHIP=$S python extensions/analysis/piece_table.py
echo ""; echo "##### error split"
python extensions/analysis/error_split.py --ckpt "$S"
echo ""; echo "##### held-out seeds, FEATK=256"
python extensions/analysis/heldout_seeds.py --dir /scratch/pmohseni/omr/candhv256 \
    --models "$M/nbr/nbrp64_s*.pt"
echo "##### done"
