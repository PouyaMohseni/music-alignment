#!/bin/bash
#SBATCH --job-name=sysslack
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/sysslack-%j.log
# Pin candidates inside the predicted SYSTEM box.
#
# 25.4% of remaining errors are "correct horizontally, jumped to a neighbouring
# system" -- the right place within a staff, the wrong staff. The system
# readout is 91.7% accurate and the harness already has a constraint that keeps
# note candidates inside the predicted system box, written as a probe and never
# once run with the learned selector in front of it.
#
# The risk is real and is why slack is swept rather than switched on: when the
# system readout is wrong, the filter removes the correct candidate entirely.
# It refuses to empty the candidate set, so a wrong box costs ranking rather
# than a no-detection frame, but a tight slack on a wrong box still hurts.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
export SCORER_PATH=/scratch/pmohseni/omr/scorer/grid/vel_p8.pt SCORER_BLEND=0.7
R=/scratch/pmohseni/omr/sysslack; mkdir -p "$R"
for S in 0 10 30 60 120; do
    export SYS_SLACK=$S REC_OUT="$R/slack$S.npz"
    echo ""; echo "##### SYS_SLACK=$S  (0 = off, the shipped 93.4)"
    python extensions/hooks/run_eval_search.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 \
      | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
      | grep -E "^<= 0.5|^Average accuracy|rror|Traceback"
done
