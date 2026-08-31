#!/bin/bash
#SBATCH --job-name=scorer-ve
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/scorer_ve-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
python - <<'PY' || { echo "ENVIRONMENT BROKEN, aborting"; exit 1; }
import numpy, cv2, torch, mpmath, sympy, scipy
print(f'[ENV] numpy {numpy.__version__} cv2 {cv2.__version__} torch {torch.__version__} ok')
PY
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
M=/scratch/pmohseni/omr/scorer; R=/scratch/pmohseni/omr/velval; mkdir -p "$R"
for V in ir_only vel_only vel_feat feat_base; do
    [ -f "$M/$V.pt" ] || { echo ""; echo "##### $V missing"; continue; }
    export SCORER_PATH=$M/$V.pt SCORER_BLEND=0.7 REC_OUT="$R/${V}_room.npz"
    echo ""; echo "##### $V blend=0.7 room"
    python extensions/hooks/run_eval_search.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |^Average|rror"
done
