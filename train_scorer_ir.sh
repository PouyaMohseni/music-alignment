#!/bin/bash
#SBATCH --job-name=scorer-ir
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=11:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/scorer_ir-%j.log
# The direct test of why the selector transfers badly. Fit on clean synthetic
# audio it recovers 36.5% of the validation headroom but LOSES on room alone
# (84.5 vs the hand prior's 86.5), because the confusions it exists to resolve
# barely occur in a training set where the detector's own argmax is already
# 92.7% correct. The reverberant dump manufactures those confusions.
#
# Selection uses the REVERBERANT validation split, since choosing on clean
# validation is what produced `big` -- best on valid (95.79) and second-worst on
# room (88.4). Room is never touched either way.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
C=/scratch/pmohseni/omr/cand
I=/scratch/pmohseni/omr/cand_ir
M=/scratch/pmohseni/omr/scorer; mkdir -p "$M"

run () { local tag=$1; shift
    if [ -f "$M/$tag.pt" ]; then echo ""; echo "########## $tag already fit, skipping"; return; fi
    echo ""; echo "########## $tag  $*"
    python extensions/analysis/train_cand_scorer.py --out "$M/$tag.pt" "$@" 2>&1 | grep -vE "^\s*$"; }

# union of clean and reverberant, selected on reverberant validation
run ir_union --train "$C/train_c*.npz" "$I/train_c*.npz" --valid "$I/valid_c0.npz"
# reverberant only, to separate "more data" from "the right data"
run ir_only  --train "$I/train_c*.npz"                    --valid "$I/valid_c0.npz"
# union again but selected on CLEAN validation, isolating the selection split
run ir_cleanval --train "$C/train_c*.npz" "$I/train_c*.npz" --valid "$C/valid_c0.npz"
echo ""; echo "Job finished at $(date)"
