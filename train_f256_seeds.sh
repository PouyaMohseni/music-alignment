#!/bin/bash
#SBATCH --job-name=f256seed
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/f256seed-%j.log
# 94.24 is also a single seed. vel_p8's own config spreads 90.79-93.42 across
# five draws, so one checkpoint says little about a configuration. Four more
# seeds of the matched-supply config, then room for all of them, so f256 is
# compared as a CONFIG MEAN against vel_p8's config mean rather than best draw
# against best draw.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf256
M=/scratch/pmohseni/omr/scorer/f256; mkdir -p "$M"
for S in 1 2 3 4; do
    T="$M/vel_p8_f256_s$S.pt"
    [ -f "$T" ] && { echo "##### seed $S present"; continue; }
    echo ""; echo "##### fitting f256 seed $S"
    python extensions/analysis/train_cand_scorer.py --out "$T" \
        --train "$F/train_c*.npz" --valid "$F/valid.npz" \
        --use_feat --featproj 8 --seed $S 2>&1 | tail -2
done
echo ""; echo "########## room, config means ##########"
python - <<'PY'
import glob, sys
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load as load_ckpt
pages = load_with_feat('/scratch/pmohseni/omr/candf256/room.npz')
M = '/scratch/pmohseni/omr/scorer'
ir, _ = rollout(load_ckpt(f'{M}/ir_only.pt')[0], pages, blend=0.7)
print(f'ir_only {ir:.2f}')
for tag, paths in (
        ('vel_p8   (trained 128)',
         [f'{M}/grid/vel_p8.pt'] + sorted(glob.glob(f'{M}/seeds/vel_p8_s*.pt'))),
        ('vel_p8_f256 (trained 256)',
         [f'{M}/f256/vel_p8_f256.pt'] + sorted(glob.glob(f'{M}/f256/vel_p8_f256_s*.pt')))):
    v = []
    for p in paths:
        acc, _ = rollout(load_ckpt(p)[0], pages, blend=0.7)
        v.append(acc)
        print(f'  {p.split("/")[-1]:24s} {acc:6.2f}', flush=True)
    v = np.array(v)
    print(f'  -> {tag}: mean {v.mean():.2f} sd {v.std(ddof=1) if len(v)>1 else 0:.2f} '
          f'best {v.max():.2f} (vs ir_only {ir:.2f})\n')
PY
