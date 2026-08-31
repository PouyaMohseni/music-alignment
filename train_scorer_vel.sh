#!/bin/bash
#SBATCH --job-name=scorer-v
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/scorer_v-%j.log
# Give the ranker a sense of SPEED, not just position.
#
# Every feature so far sees exactly one past position, so the model knows where
# the music was and not how fast it is moving. 60.8% of remaining errors are
# timing drift within two bars, which is precisely what a velocity estimate is
# for. Four new features, all PER-CANDIDATE (each box has its own offset from
# the constant-velocity extrapolation) -- the property that made backbone
# features work and made z useless.
#
# Not the constant-velocity decode that failed on v13: that BLENDED an
# extrapolation into the observation every frame and cost precision everywhere.
# Here it is a feature the model may weight or ignore, and vel_off is the
# control that says which happened.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
I=/scratch/pmohseni/omr/cand_ir
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer

run () { local tag=$1; shift
    [ -f "$M/$tag.pt" ] && { echo ""; echo "########## $tag already fit"; return; }
    echo ""; echo "########## $tag  $*"
    python extensions/analysis/train_cand_scorer.py --out "$M/$tag.pt" "$@" 2>&1 \
        | stdbuf -oL grep --line-buffered -vE "^\s*$"; }

# velocity alone, against the same recipe that produced the shipped 91.4
run vel_only  --train "$I/train_c*.npz" --valid "$I/valid_c0.npz"
# velocity together with the backbone features that just gained +10.7 headroom
run vel_feat  --train "$F/train_c*.npz" --valid "$F/valid.npz" --use_feat
echo ""; echo "Job finished at $(date)"
