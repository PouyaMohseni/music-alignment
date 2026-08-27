#!/bin/bash
#SBATCH --job-name=blend
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/blend-%j.log
# The learned selector LOSES alone (84.5 vs the hand prior's 86.5) and the hand
# prior loses to the mixture (89.7 at blend=0.5). Three consecutive blend values
# beat both endpoints (89.4 / 89.7 / 89.1), so the two scores carry different
# information rather than one being a noisy copy of the other.
#
# Caveat this sweep has to respect: the learned term is an untethered logit and
# the hand term is a log-probability, so `blend` sets a RELATIVE SCALE as much
# as a mixing weight, and its optimum partly reflects that. A flat top would say
# the mixture is robust; a sharp spike would say we are tuning a scale factor on
# the test set. Map it finely enough to tell those apart.
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
M=/scratch/pmohseni/omr/scorer
REC=/scratch/pmohseni/omr/blend; mkdir -p "$REC"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5
export CLUSTER_PX=0 ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { local tag=$1 tier=$2
    export REC_OUT="$REC/${tag}_${tier}.npz"
    echo ""; echo "##### $tag  tier=$tier  $(basename "$SCORER_PATH") blend=$SCORER_BLEND"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/${tier}_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$"; }

echo "=== A. fine blend grid on room (is the top flat or a spike?) ==="
export SCORER_PATH=$M/base.pt
for B in 0.15 0.3 0.4 0.45 0.55 0.6 0.7 0.9; do
    export SCORER_BLEND=$B; run "base_b$B" room
done

echo ""; echo "=== B. the other variants at the mixture that works ==="
export SCORER_BLEND=0.5
for V in noabs big bignoise strongnoise longer; do
    [ -f "$M/$V.pt" ] || { echo ""; echo "##### $V: no checkpoint yet, skipped"; continue; }
    export SCORER_PATH=$M/$V.pt; run "${V}_b0.5" room
done

echo ""; echo "=== C. does the mixture hold on tiers it was never tuned on? ==="
export SCORER_PATH=$M/base.pt SCORER_BLEND=0.5
run base_b0.5 do
run base_b0.5 rp_synth
echo ""; echo "Job finished at $(date)"
