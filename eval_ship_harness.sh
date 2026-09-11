#!/bin/bash
#SBATCH --job-name=shipharn
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/shipharn-%j.log
# The demo panels for the checkpoint the pages now report.
#
# nbrp64_s0 is what the selection rule picks (config by held-out mean, draw by
# held-out score, room read once) and it scored 94.9 through the harness. The
# panels still show vel_p8's path, so this records nbrp64_s0's frame-by-frame
# trajectory in the SAME harness run that has to reproduce 94.9, then builds
# the payload from it. The case list is chosen by rule inside the builder.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0 ffmpeg 2>/dev/null || module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
which ffmpeg || { echo "no ffmpeg, cannot cut excerpts"; exit 1; }
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
export SCORER_PATH=/scratch/pmohseni/omr/scorer/nbr/nbrp64_s0.pt SCORER_BLEND=0.7
T=/scratch/pmohseni/omr/traj
export REC_OUT="$T/nbrp64s0_room.rec.npz" TRAJ_OUT="$T/nbrp64s0_room.traj.npz"
echo "##### harness, must print 94.9"
python extensions/hooks/run_eval_search.py \
    --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
    --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
    --only_onsets 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
  | grep -E "^<= |rror|Traceback|TRAJ"
[ -f "$TRAJ_OUT" ] || { echo "NO TRAJECTORY, not building the payload"; exit 1; }
echo ""; echo "##### payload"
OURS_TRAJ="$TRAJ_OUT" DEMO_OUT=/scratch/pmohseni/omr/demo_nbrp64 \
    python extensions/analysis/build_demo_payload.py
