#!/bin/bash
#SBATCH --job-name=valsel
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/valsel-%j.log
# Choose the configuration WITHOUT looking at room.
#
# The 89.7 headline rests on two choices made after seeing room numbers: which
# of five trained scorers to use (they span 86.9-89.7 at the same blend) and
# which blend. That makes the number the top of a distribution rather than an
# estimate of the method. Fix it by selecting both on a held-out VALIDATION
# split -- the 19 msmd_valid pieces, which no scorer was trained on and which
# room evaluation never touches.
#
# Validation is run in TWO acoustic conditions because the choice of validation
# set is itself a hypothesis under test: clean synthetic validation already
# anti-selected once (`big` scored best there at 95.79 and second-worst on room
# at 88.4). Reverberant validation shares the failure mode with room, so it
# should rank variants better. Running both says whether it does.
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
REC=/scratch/pmohseni/omr/valsel; mkdir -p "$REC"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5
export CLUSTER_PX=0 ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
export IR_SEED=7 IR_PROB=1.0     # seed 7: a different room draw from the training dump

run () { export REC_OUT="$REC/$1.npz"
    echo ""; echo "##### $1  scorer=$(basename "$SCORER_PATH") blend=$SCORER_BLEND ir=${IR_PATH:-none}"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_valid" --split_files "$DATA/split_files/valid_c0_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |^Average accuracy|rror|Traceback"; }

for COND in reverberant clean; do
  if [ "$COND" = reverberant ]; then export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey
  else unset IR_PATH; fi
  echo ""; echo "############### VALIDATION, $COND ###############"
  export SCORER_PATH=$M/base.pt SCORER_BLEND=0.0; run "${COND}_handonly"
  for V in base noabs big bignoise strongnoise longer ir_union ir_only ir_cleanval; do
    [ -f "$M/$V.pt" ] || continue
    export SCORER_PATH=$M/$V.pt
    for B in 0.3 0.5 0.7; do export SCORER_BLEND=$B; run "${COND}_${V}_b${B}"; done
  done
done
echo ""; echo "Job finished at $(date)"
