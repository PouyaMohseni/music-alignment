#!/bin/bash
#SBATCH --job-name=featvel
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/featvel-%j.log
# Does the harness agree with the offline rollout once features are actually
# passed? Three arms, each with a PREDICTION that can falsify something:
#
#   ir_only   featdim=0, untouched by the fix   -> must stay 91.4 exactly
#   feat_wide claimed 94.0 with a dead branch   -> should FALL to ~91.9
#   vel_p8    the `do` proxy's pick             -> should RISE to ~92.6
#
# ir_only is the control that says the fix broke nothing; feat_wide is the one
# that confirms the retraction is real rather than a story; vel_p8 is the only
# number we actually want. If feat_wide does not fall, my diagnosis is wrong.
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
M=/scratch/pmohseni/omr/scorer; R=/scratch/pmohseni/omr/featfix; mkdir -p "$R"

python - <<'PY' || { echo "ENVIRONMENT BROKEN, aborting"; exit 1; }
import numpy, cv2, torch, mpmath, sympy, scipy
print(f'[ENV] numpy {numpy.__version__} cv2 {cv2.__version__} torch {torch.__version__}')
PY

run () {   # name  checkpoint
    local V=$1 P=$2
    [ -f "$P" ] || { echo ""; echo "##### $V MISSING $P"; return; }
    export SCORER_PATH=$P SCORER_BLEND=0.7 REC_OUT="$R/${V}_room.npz"
    echo ""; echo "##### $V blend=0.7 room  ($P)"
    python extensions/hooks/run_eval_search.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 \
      | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
      | grep -E "^<= |^Average|rror|Traceback|FEAT|SCORER"
    # an npz with zero arrays is still a valid, non-empty file, so `-s` once let
    # silently-skipped arms look like successful ones. The first version of this
    # guard demanded >1MB and cried wolf on every arm: an --only_onsets dump of
    # all 16 room pieces is ~26KB, because it holds 4149 onsets and not frames.
    # Count the actual rows instead of guessing from the file size.
    python - "$R/${V}_room.npz" <<'PY' || echo "!!!!! $V produced no usable dump"
import sys, numpy as np
d = np.load(sys.argv[1])
n = sum(d[k].size for k in d.files if k.endswith('frame_diff'))
print(f'[DUMP] {len(d.files)} arrays, {n} onsets')
sys.exit(0 if n > 1000 else 1)
PY
}

run vel_p8    "$M/grid/vel_p8.pt"
