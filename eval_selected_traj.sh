#!/bin/bash
#SBATCH --job-name=sel-traj
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/sel_traj-%j.log
# Trajectories for the shipped 91.4 configuration. eval_selected.sh recorded
# per-frame ERRORS but not the decoded PATH, so the demo cannot draw the cursor
# for the model we actually report. Same config, TRAJ_OUT on, control included
# so 86.5 and 91.4 both have to reappear.
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
python - <<'PY' || { echo "ENVIRONMENT BROKEN"; exit 1; }
import numpy, cv2, torch, mpmath, sympy
print(f'[ENV] numpy {numpy.__version__} cv2 {cv2.__version__} torch {torch.__version__} ok')
PY
CKPT=$CY/trained_models/cyolo_sb/best_model.pt
M=/scratch/pmohseni/omr/scorer
T=/scratch/pmohseni/omr/traj; mkdir -p "$T"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0 SCORER_PATH=$M/ir_only.pt

run () { export SCORER_BLEND=$2 REC_OUT="$T/$1_room.rec.npz" TRAJ_OUT="$T/$1_room.traj.npz"
    echo ""; echo "##### $1  blend=$2"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |^Average|\[TRAJ\]|rror"; }

run handonly 0.0      # must return 86.5
run selected 0.7      # must return 91.4
echo ""; echo "Job finished at $(date)"
