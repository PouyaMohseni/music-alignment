#!/bin/bash
#SBATCH --job-name=seedvar
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/seedvar-%j.log
# Is vel_p8's edge a model or a seed?
#
# vel_p8 beats ir_only by +2.0 on room with a CI of [-0.95, +3.62], and it was
# picked out of 14 candidates by a proxy. Nobody has retrained it. If refitting
# the SAME configuration on the SAME data with a different seed moves room by a
# comparable amount, then +2.0 is training noise and the selection was picking
# a lucky draw rather than a better model -- which would also explain a proxy
# correlation of only +0.407.
#
# Five seeds of the identical config. The spread across them is the yardstick
# every claimed difference between checkpoints has to clear.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/seeds; mkdir -p "$M"

for S in 1 2 3 4 5; do
    T="$M/vel_p8_s$S.pt"
    [ -f "$T" ] && { echo "########## seed $S already fit"; continue; }
    echo ""; echo "########## fitting vel_p8 seed $S"
    python extensions/analysis/train_cand_scorer.py --out "$T" \
        --train "$F/train_c*.npz" --valid "$F/valid.npz" \
        --use_feat --featproj 8 --seed $S 2>&1 \
      | stdbuf -oL grep --line-buffered -vE "^\s*$" | tail -3
done

echo ""; echo "########## room + do for every seed (offline rollout) ##########"
python - <<'PY'
import glob, sys
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load as load_ckpt
room = load_with_feat('/scratch/pmohseni/omr/candf/room.npz')
do   = load_with_feat('/scratch/pmohseni/omr/candf/do.npz')
rows = []
for p in ['/scratch/pmohseni/omr/scorer/ir_only.pt',
          '/scratch/pmohseni/omr/scorer/grid/vel_p8.pt'] + \
         sorted(glob.glob('/scratch/pmohseni/omr/scorer/seeds/vel_p8_s*.pt')):
    m = load_ckpt(p)[0]
    r, _ = rollout(m, room, blend=0.7)
    d, _ = rollout(m, do,   blend=0.7)
    rows.append((p.split('/')[-1], d, r))
    print(f'{rows[-1][0]:20s} do {d:6.2f}   room {r:6.2f}', flush=True)
seeds = [r for r in rows if '_s' in r[0]]
if seeds:
    v = np.array([r[2] for r in seeds])
    print(f'\nseed spread on room: min {v.min():.2f}  max {v.max():.2f}  '
          f'sd {v.std(ddof=1):.2f}  range {v.max() - v.min():.2f}')
    ir = [r[2] for r in rows if r[0].startswith('ir_only')][0]
    print(f'ir_only room = {ir:.2f}; vel_p8 original = '
          f'{[r[2] for r in rows if r[0] == "vel_p8.pt"][0]:.2f}')
    gap = abs(float(np.mean(v)) - ir)
    rng = float(v.max() - v.min())
    verdict = ('seed spread exceeds the vel_p8 - ir_only gap: the gap is noise'
               if rng >= gap else 'gap survives seed variation')
    print(f'\nmean seed room {float(np.mean(v)):.2f}, gap to ir_only {gap:+.2f}, '
          f'seed range {rng:.2f}')
    print('VERDICT: ' + verdict)
PY
