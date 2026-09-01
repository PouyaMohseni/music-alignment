#!/bin/bash
#SBATCH --job-name=back-hand
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/back_hand-%j.log
# Price backward motion properly, on the HAND decoder so the effect is not
# diluted by the learned selector.
#
# Measured on room: ground truth steps backwards on 0.73% of onsets, our decoder
# on 4.50% -- six times too often -- and those are its worst steps (35.7%
# correct in the 50-200 px band against 88.3% for a normal forward one). The
# cause is that the prior's jump floor was symmetric: 400 px backwards priced
# exactly like 400 px forwards.
#
# The -1e6 arm is the extreme: backward motion effectively forbidden, so the
# tracker is strictly monotone. Ground truth is monotone to 0.73%, and the
# performances take no repeats at all -- every notehead sounds exactly once --
# so a hard constraint is defensible ON THIS DATA and would not be on a
# performance that takes its repeats.
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
REC=/scratch/pmohseni/omr/backhand; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_CLASSES=0 C2_TOPK=100000
export C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { export REC_OUT="$REC/$1.npz" TRAJ_OUT="$REC/$1.traj.npz"
    echo ""; echo "##### $1  back_logp=${C2_BACK:-symmetric(-6)}"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |^Average|rror|Traceback"; }

unset C2_BACK; run ctrl                       # must return 86.5
for B in -8 -12 -20 -40 -1000000; do export C2_BACK=$B; run "b$B"; done
echo ""; echo "Job finished at $(date)"
