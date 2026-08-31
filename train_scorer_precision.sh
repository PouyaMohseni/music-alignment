#!/bin/bash
#SBATCH --job-name=precise
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/precise-%j.log
# Aim at the threshold we have never aimed at.
#
# Room, hand decode -> selector:  0.05 s  63.8 -> 72.2   ceiling 94.0
#                                 0.50 s  86.5 -> 91.4   ceiling 96.0
#
# The selector already helps MORE at 0.05 s (+8.4) than at 0.5 s (+4.9), which
# I had assumed the opposite of. And the gap left there is 22 points against
# 4.6. So the tight column is both the larger prize and the one responding
# better, while nothing in the training objective is pointed at it: the soft
# label exp(-|dt|/tau) with tau=3 frames rewards landing within ~0.15 s, and
# epoch selection uses the 0.5 s rollout.
#
# tau tightens the target; sel_th=1 selects the epoch on the 0.05 s rollout.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
I=/scratch/pmohseni/omr/cand_ir
M=/scratch/pmohseni/omr/scorer

run () { local tag=$1; shift
    [ -f "$M/$tag.pt" ] && { echo ""; echo "########## $tag already fit"; return; }
    echo ""; echo "########## $tag  $*"
    python extensions/analysis/train_cand_scorer.py --out "$M/$tag.pt" \
        --train "$I/train_c*.npz" --valid "$I/valid_c0.npz" "$@" 2>&1 | stdbuf -oL grep --line-buffered -vE "^\s*$"; }

run prec_tau1   --tau 1.0 --sel_th 1.0
run prec_tau05  --tau 0.5 --sel_th 1.0
run prec_tau2   --tau 2.0 --sel_th 1.0
run prec_th_only          --sel_th 1.0     # tau unchanged: isolates the epoch-selection change
echo ""; echo "Job finished at $(date)"
