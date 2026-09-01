#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/grid-%j.log
# A clean grid, because the last one was confounded by my own mid-flight edit.
#
# feat_wide (94.0 on room) turned out to have nf=24 -- it picked up the velocity
# features committed at 15:55 while the grid was still running -- while
# feat_base and feat_small have nf=20. So feat_wide differs from feat_base in
# TWO ways, and the two clean comparisons that survive point opposite ways:
#
#   with velocity     proj 32 -> 64   90.0 -> 94.0   (+4.0)
#   without velocity  proj  8 -> 32   91.3 -> 88.9   (-2.4)
#
# {no_vel, vel} x {proj 8, 32, 64}, feature set frozen, one dump, one code
# version. --no_vel truncates to the first 20 features, which is EXACTLY the old
# set because FEATURE_NAMES only ever grows at the end.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/grid; mkdir -p "$M"

run () { local tag=$1; shift
    [ -f "$M/$tag.pt" ] && { echo ""; echo "########## $tag already fit"; return; }
    echo ""; echo "########## $tag  $*"
    python extensions/analysis/train_cand_scorer.py --out "$M/$tag.pt" \
        --train "$F/train_c*.npz" --valid "$F/valid.npz" --use_feat "$@" 2>&1 \
        | stdbuf -oL grep --line-buffered -vE "^\s*$"; }

for P in 8 32 64; do
    run "novel_p$P" --no_vel --featproj $P
    run "vel_p$P"            --featproj $P
done
echo ""; echo "Job finished at $(date)"
