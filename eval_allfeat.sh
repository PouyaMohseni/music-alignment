#!/bin/bash
#SBATCH --job-name=allfeat
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/allfeat-%j.log
# Every feature model's room number was measured with a dead feature branch,
# not just feat_wide's. Only ir_only (featdim=0), feat_wide and vel_p8 have
# valid numbers now, so "vel_p8 is best" rests on a table that is mostly
# invalid. Re-measure the whole set through the fixed harness before believing
# the ranking.
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
M=/scratch/pmohseni/omr/scorer; R=/scratch/pmohseni/omr/allfeat; mkdir -p "$R"

python - <<'PY' || { echo "ENVIRONMENT BROKEN, aborting"; exit 1; }
import numpy, cv2, torch, mpmath, sympy, scipy
print(f'[ENV] numpy {numpy.__version__} torch {torch.__version__}')
PY

run () {
    local V=$1 P=$2
    [ -f "$P" ] || { echo ""; echo "##### $V MISSING"; return; }
    export SCORER_PATH=$P SCORER_BLEND=0.7 REC_OUT="$R/${V}_room.npz"
    echo ""; echo "##### $V"
    python extensions/hooks/run_eval_search.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 \
      | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
      | grep -E "^<= 0.5|rror|Traceback"
}

run ir_only    "$M/ir_only.pt"          # control, featdim=0, must stay 91.4
for V in feat_base feat_small feat_wide vel_feat vel_only noz_only; do
    run "$V" "$M/$V.pt"
done
for V in vel_p8 vel_p32 vel_p64 novel_p8 novel_p32 novel_p64; do
    run "$V" "$M/grid/$V.pt"
done
