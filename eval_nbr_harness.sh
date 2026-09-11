#!/bin/bash
#SBATCH --job-name=nbrharn
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/nbrharness.log
# Do the offline nbr numbers survive the real harness?
#
# Every nbr figure so far comes from the offline rollout. The nine neighbourhood
# features are computed by build() in BOTH paths, but through different callers
# -- blend_rollout offline, ScorerDecoder in the harness -- and "a feature that
# is present in one path and absent or different in the other" is exactly the
# bug that has now bitten this project three times (the dead feature branch,
# zeroed pitch columns, mismatched audio). So before any nbr config mean is
# trusted, three checkpoints go through the harness and must match offline:
#
#   vel_p8     the control, must reproduce 93.42
#   nbr_s6     proj 8's held-out pick,  offline 92.91
#   nbrp64_s0  proj 64's held-out pick, offline 94.89
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
M=/scratch/pmohseni/omr/scorer; R=/scratch/pmohseni/omr/nbrharness; mkdir -p "$R"
python - <<'PY' || { echo "ENVIRONMENT BROKEN, aborting"; exit 1; }
import numpy, cv2, torch, mpmath, sympy, scipy
print(f'[ENV] numpy {numpy.__version__} torch {torch.__version__}')
PY
run () {   # name  checkpoint  offline-number
    local V=$1 P=$2 WANT=$3
    [ -f "$P" ] || { echo "##### $V MISSING $P"; return; }
    export SCORER_PATH=$P SCORER_BLEND=0.7 REC_OUT="$R/${V}_room.npz"
    echo "##### $V   (offline rollout said $WANT)"
    python extensions/hooks/run_eval_search.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 \
      | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
      | grep -E "^<= 0.5|rror|Traceback"
}
run vel_p8     "$M/grid/vel_p8.pt"     93.42
run nbr_s6     "$M/nbr/nbr_s6.pt"      92.91
run nbrp64_s0  "$M/nbr/nbrp64_s0.pt"   94.89
echo "##### done"
