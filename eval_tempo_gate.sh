#!/bin/bash
#SBATCH --job-name=tgate
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/tgate-%j.log
# blend_rollout underpins nearly every analysis in this project, and it now
# threads a tracked tempo through build(). Models fitted before the tempo
# features have model.nf <= 33, so the four new columns are truncated away and
# their numbers MUST be unchanged. If they are not, something else moved.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
import sys, torch, numpy as np
torch.set_num_threads(1)
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load as load_ckpt
from extensions.heads.cand_features import NF
print(f'NF is now {NF}')
pages = load_with_feat('/scratch/pmohseni/omr/candf256/room.npz')
GATES = [('ir_only', '/scratch/pmohseni/omr/scorer/ir_only.pt', 0.7, 91.44),
         ('vel_p8',  '/scratch/pmohseni/omr/scorer/grid/vel_p8.pt', 0.7, 93.42),
         ('vel_p8 blend0', '/scratch/pmohseni/omr/scorer/grid/vel_p8.pt', 0.0, 86.50)]
bad = 0
for name, p, b, want in GATES:
    m = load_ckpt(p)[0]
    got, _ = rollout(m, pages, blend=b)
    ok = abs(got - want) < 0.05
    bad += not ok
    print(f'  {name:16s} nf={m.nf:3d} got {got:6.2f} want {want:6.2f}  '
          f'{"OK" if ok else "CHANGED"}')
sys.exit(1 if bad else 0)
PY
