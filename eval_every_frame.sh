#!/bin/bash
#SBATCH --job-name=everyfr
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/everyfr%a-%A.log
# Track EVERY frame, not only the annotated onsets.
#
# --only_onsets makes cyolo's load_dataset drop every frame that is not an
# annotated note onset, so the tracker only runs at the moments notes start --
# and our decoder carries its state from one of those moments to the next and
# scales its prior (and its velocity features) by the gap between them. Those
# gaps are annotated inter-onset intervals: timing that music without a
# symbolic reference would not provide. Without the flag every frame is loaded,
# decoded and scored; the trajectory is recorded so every_frame.py can ALSO
# score just the onset frames of the same run (tracked on every frame, judged
# where the notes are).
#
#   0 baseline  cyolo_sb's own argmax -- no history, so no gaps to exploit
#   1 hand      zero-parameter transition prior
#   2 flat      featureless selector ir_only, blend 0.7
#   3 ship      nbrp64_s0, blend 0.7 -- the reported model
#
# Settings are copied from the onset-only runs that gave 79.97 / 86.5 / 91.4 /
# 94.9 (eval_trajectories.sh, eval_selected_traj.sh, eval_ship_harness.sh);
# the only change is the missing --only_onsets.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
M=/scratch/pmohseni/omr/scorer
T=/scratch/pmohseni/omr/everyframe; mkdir -p "$T"
export C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0 CLUSTER_PX=0
export TIME_SIG_POW=0 TIME_REF=5 ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
case ${SLURM_ARRAY_TASK_ID} in
  0) A=baseline; export SEARCH_KIND=beam BEAM=1 C2_TOPK=100000 C2_CLASSES='' TIME_MU_POW=0 ;;
  1) A=hand;     export SEARCH_KIND=beam BEAM=1 C2_TOPK=100000 C2_CLASSES='0' TIME_MU_POW=1 ;;
  2) A=flat;     export SEARCH_KIND=scorer C2_TOPK=256 TIME_MU_POW=1 \
                        SCORER_PATH=$M/ir_only.pt SCORER_BLEND=0.7 ;;
  3) A=ship;     export SEARCH_KIND=scorer C2_TOPK=256 TIME_MU_POW=1 \
                        SCORER_PATH=$M/nbr/nbrp64_s0.pt SCORER_BLEND=0.7 ;;
esac
export REC_OUT="$T/${A}_room.rec.npz" TRAJ_OUT="$T/${A}_room.traj.npz"
echo "##### $A, every frame (no --only_onsets)   $(date)"
python extensions/hooks/run_eval_search.py \
    --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
    --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
  | grep -E "^<= |rror|Traceback|TRAJ|REC"
[ -f "$TRAJ_OUT" ] || { echo "NO TRAJECTORY"; exit 1; }
echo ""
python extensions/analysis/every_frame.py "$TRAJ_OUT"
echo "##### done $(date)"
