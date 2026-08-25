#!/bin/bash
#SBATCH --job-name=train-scorer
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/train_scorer-%j.log
# Fit the selector on the 353-piece TRAIN split, select on the 19-piece VALID
# split, never touch room. Variants exist to measure things rather than to
# search: noabs asks how much the model leans on absolute objectness (which
# shifts between synthetic training audio and real room test audio), nonoise
# asks whether the previous-position noise that stands in for exposure bias is
# doing anything, big asks whether 9.7k parameters is the binding constraint.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
C=/scratch/pmohseni/omr/cand
M=/scratch/pmohseni/omr/scorer; mkdir -p "$M"

run () { local tag=$1; shift
    echo ""; echo "########## $tag  $*"
    python extensions/analysis/train_cand_scorer.py \
        --train "$C/train_c*.npz" --valid "$C/valid_c0.npz" \
        --out "$M/$tag.pt" "$@" 2>&1 | grep -vE "^\s*$"; }

run base
run noabs        --no_abs_obj
run nonoise      --noise_p 0
run strongnoise  --noise_p 0.5 --noise_px 60
run big          --hidden 128 --embed 64
echo ""; echo "Job finished at $(date)"
