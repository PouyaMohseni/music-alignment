#!/bin/bash
#SBATCH --job-name=seedf256
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/seedf256-%j.log
# seedvar trained 4 seeds then hit its wall before evaluating any of them.
# Its training-time headroom metric ran 54.8 / 77.4 / 82.6 / 84.3 percent
# across seeds, a spread wide enough that the room gap between checkpoints may
# be a draw rather than a model. Evaluate them.
#
# Both feature supplies, because they are now known to matter: candf caps
# features at the top 128 (what training saw), candf256 supplies all 256 (what
# the harness actually does).
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
import glob, sys
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load as load_ckpt
M = '/scratch/pmohseni/omr/scorer'
mods = [('ir_only', f'{M}/ir_only.pt'), ('vel_p8', f'{M}/grid/vel_p8.pt'),
        ('vel_p8_f256', f'{M}/f256/vel_p8_f256.pt')] + \
       [(p.split('/')[-1][:-3], p) for p in
        sorted(glob.glob(f'{M}/seeds/vel_p8_s*.pt'))]
mods = [(n, p) for n, p in mods if glob.glob(p)]
res = {}
for tag, dump in (('candf(128)', '/scratch/pmohseni/omr/candf/room.npz'),
                  ('candf256', '/scratch/pmohseni/omr/candf256/room.npz')):
    pages = load_with_feat(dump)
    for n, p in mods:
        acc, _ = rollout(load_ckpt(p)[0], pages, blend=0.7)
        res.setdefault(n, {})[tag] = acc
    del pages
print(f'{"model":14s} {"candf(128)":>11s} {"candf256":>9s}')
for n, _ in mods:
    print(f'{n:14s} {res[n]["candf(128)"]:11.2f} {res[n]["candf256"]:9.2f}')
s = [res[n]['candf256'] for n, _ in mods if '_s' in n]
if len(s) > 1:
    s = np.array(s)
    ir = res['ir_only']['candf256']; v8 = res['vel_p8']['candf256']
    print(f'\nseeds on candf256: n={len(s)} min {s.min():.2f} max {s.max():.2f} '
          f'sd {s.std(ddof=1):.2f} range {s.max()-s.min():.2f}')
    print(f'ir_only {ir:.2f}   vel_p8 (seed 0) {v8:.2f}   seed mean {s.mean():.2f}')
    print(f'\nSEED RANGE {s.max()-s.min():.2f} vs vel_p8-minus-ir_only '
          f'{v8-ir:+.2f}')
    print('VERDICT: ' + ('seed noise is as large as the gap; the specific '
                         'checkpoint is a draw, though the CONFIG may still win'
                         if (s.max()-s.min()) >= abs(v8-ir) else
                         'gap exceeds seed spread'))
PY
