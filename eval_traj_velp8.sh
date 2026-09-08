#!/bin/bash
#SBATCH --job-name=trajv8
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/trajv8-%j.log
# Frame-by-frame positions for vel_p8, so the demo panels show the model we
# actually ship rather than the 91.4 ir_only run they were built from.
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
T=/scratch/pmohseni/omr/traj
export REC_OUT="$T/velp8_room.rec.npz" TRAJ_OUT="$T/velp8_room.traj.npz"
python extensions/hooks/run_eval_search.py \
    --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
    --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
    --only_onsets 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
  | grep -E "^<= |rror|Traceback|TRAJ"
ls -l "$T/velp8_room.traj.npz"
