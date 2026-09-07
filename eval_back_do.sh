#!/bin/bash
#SBATCH --job-name=back-do
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/back_do-%j.log
# Validate back_logp on a tier it was NOT chosen on.
#
# The sweep on room gave -6 (shipped) 86.5, -8 87.4, -12 86.4, and 81.9 from
# -20 down through hard monotone. -8 was picked on room, which is the selection
# problem I have been removing everywhere else. `do` is real audio and the only
# proxy with signal against room (Spearman +0.60), so if -8 is also the peak
# there it is a property of the decoder rather than of the room draw.
#
# Backward motion is not simply bad: 56.4% of the >200 px backward steps land
# correctly and look like recoveries from an earlier wrong commit, which is why
# forbidding it costs 4.6 points.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
CKPT=$CY/trained_models/cyolo_sb/best_model.pt
REC=/scratch/pmohseni/omr/backdo; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_CLASSES=0 C2_TOPK=100000
export C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { export REC_OUT="$REC/$1_$2.npz"
    echo ""; echo "##### tier=$2  back=${C2_BACK:-symmetric(-6)}"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/$2_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |rror|Traceback"; }

for TIER in do rp_synth; do
    unset C2_BACK; run ctrl "$TIER"
    for B in -8 -12 -20; do export C2_BACK=$B; run "b$B" "$TIER"; done
    unset C2_BACK
done
echo ""; echo "Job finished at $(date)"
