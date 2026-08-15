#!/bin/bash
#SBATCH --job-name=c2-fitted
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/c2_fitted-%j.log
# C2 with the motion model MEASURED from the 353 training pieces instead of
# guessed. Truth: median 4.80 px/frame, std 4.96. C2 shipped fwd=6.0, sigma=18.0
# -- a prior ~3.6x too wide to constrain anything.
#
# The PRE-REGISTERED run is fwd=4.8 sigma=5.0: the measured values, chosen
# before seeing any test number. The rest is context, and picking a winner from
# it would be test-set tuning, so it is reported as a sensitivity curve.
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
REC=/scratch/pmohseni/omr/c2_fitted; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1

run () {  # tag fwd sigma
    export C2_FWD=$2 C2_SIGMA=$3 REC_OUT="$REC/$1_room.npz"
    echo ""; echo "########## $1  (fwd=$2 sigma=$3) ##########"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]" | tail -10
}
run measured  4.8  5.0      # <-- pre-registered
run shipped   6.0 18.0      # control, must reproduce 84.7
run s8        4.8  8.0
run s12       4.8 12.0
run f6s5      6.0  5.0
run s3        4.8  3.0
echo ""; echo "Job finished at $(date)"
