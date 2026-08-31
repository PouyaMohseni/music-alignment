#!/bin/bash
#SBATCH --job-name=precise-e
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/precise_eval-%j.log
# The precision-targeted selectors on room. Selection among them happens on the
# reverberant validation split inside the training job; this reports the full
# threshold table so the tight columns are visible next to 0.5 s.
#
# The reference to beat, from the configuration already selected on validation:
#   0.05 s  72.2    0.1 s  74.1    0.5 s  91.4
# and the causal ceilings:
#   0.05 s  94.0    0.1 s  94.6    0.5 s  96.0
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
REC=/scratch/pmohseni/omr/precise; mkdir -p "$REC"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { local v=$1 b=$2
    [ -f "$M/$v.pt" ] || { echo ""; echo "##### $v: no checkpoint, skipped"; return; }
    export SCORER_PATH=$M/$v.pt SCORER_BLEND=$b REC_OUT="$REC/${v}_b${b}_room.npz"
    echo ""; echo "##### $v  blend=$b  tier=room"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |^Average|rror"; }

# reference: the configuration validation already chose, so every row below is
# read against a number produced by the identical code path
run ir_only 0.7
for V in prec_tau1 prec_tau05 prec_tau2 prec_th_only; do
    run "$V" 0.7
done
# the best of them also at a lower blend, in case a precision-tuned score wants
# less weight against the prior
run prec_tau1 0.5
echo ""; echo "Job finished at $(date)"
