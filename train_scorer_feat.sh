#!/bin/bash
#SBATCH --job-name=scorer-f
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/scorer_f-%j.log
# Re-fit the ranking function on the SAME input it already has.
#
# Detect.m[0] is 1,935 parameters mapping a 128-dim feature per grid cell to
# objectness. It was trained to DETECT; nothing ever trained it to rank against
# a temporal prior. These are those exact 128 numbers, so a head over them is
# the most direct possible attack on the ranking -- and unlike z, which is one
# vector per FRAME and turned out to be useless, features differ BETWEEN
# candidates, which is the only way to separate two boxes the geometry cannot.
#
# noz_only is the control to beat: 95.87 valid, 55.9% of headroom, 91.0 room.
# Same data, same recipe, features withheld.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer

run () { local tag=$1; shift
    [ -f "$M/$tag.pt" ] && { echo ""; echo "########## $tag already fit"; return; }
    echo ""; echo "########## $tag  $*"
    python extensions/analysis/train_cand_scorer.py --out "$M/$tag.pt" \
        --train "$F/train_c*.npz" --valid "$F/valid.npz" "$@" 2>&1 | stdbuf -oL grep --line-buffered -vE "^\s*$"; }

run nofeat_ctrl                                  # control on the SAME dump
run feat_base   --use_feat
run feat_small  --use_feat --featproj 8          # narrower path, less to overfit
echo ""; echo "Job finished at $(date)"
