#!/bin/bash
#SBATCH --job-name=scorer-z
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=11:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/scorer_z-%j.log
# Does giving the selector the detector's own audio vector help?
#
# noz_union is the control and it matters: it trains on the SAME re-dumped data
# with z withheld, so any difference is attributable to z rather than to the
# dump having been regenerated.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
Z=/scratch/pmohseni/omr/candz
ZI=/scratch/pmohseni/omr/candz_ir
M=/scratch/pmohseni/omr/scorer

run () { local tag=$1; shift
    [ -f "$M/$tag.pt" ] && { echo ""; echo "########## $tag already fit"; return; }
    echo ""; echo "########## $tag  $*"
    python extensions/analysis/train_cand_scorer.py --out "$M/$tag.pt" "$@" 2>&1 | grep -vE "^\s*$"; }

run noz_union --train "$Z/train_c*.npz" "$ZI/train_c*.npz" --valid "$ZI/valid.npz"
run z_union   --train "$Z/train_c*.npz" "$ZI/train_c*.npz" --valid "$ZI/valid.npz" --use_z
run z_only    --train "$ZI/train_c*.npz"                   --valid "$ZI/valid.npz" --use_z
echo ""; echo "Job finished at $(date)"
