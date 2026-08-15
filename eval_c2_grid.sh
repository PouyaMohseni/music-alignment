#!/bin/bash
#SBATCH --job-name=c2-grid
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/c2_grid-%j.log
# The sigma curve was still RISING at 18.0 -- the shipped constant, and the
# largest value ever tested. It must turn over eventually (sigma -> inf makes
# the prior flat, degenerating to plain argmax = 79.9), so find the peak.
# jump_logp has never been swept at all, and on the "the prior models TRACKER
# uncertainty, not player motion" reading it is the recovery escape hatch --
# arguably the more important constant of the two.
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
REC=/scratch/pmohseni/omr/c2_grid; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1

run () {  # tag fwd sigma lam jump
    export C2_FWD=$2 C2_SIGMA=$3 C2_LAM=$4 C2_JUMP=$5 REC_OUT="$REC/$1_room.npz"
    echo ""; echo "##### $1 fwd=$2 sigma=$3 lam=$4 jump=$5"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | grep -E "^<= 0.5|^Average accuracy for Bar"
}

# 1. push sigma past the shipped value
run sig18  6.0 18.0 1.0 -6.0     # control = 84.7
run sig24  6.0 24.0 1.0 -6.0
run sig30  6.0 30.0 1.0 -6.0
run sig45  6.0 45.0 1.0 -6.0
run sig60  6.0 60.0 1.0 -6.0
run sig90  6.0 90.0 1.0 -6.0
# 2. the escape hatch, never swept
run jm3    6.0 18.0 1.0 -3.0
run jm4p5  6.0 18.0 1.0 -4.5
run jm9    6.0 18.0 1.0 -9.0
run jm12   6.0 18.0 1.0 -12.0
# 3. weight on the transition term, re-swept on the FIXED decoder
run lam0p5 6.0 18.0 0.5 -6.0
run lam1p5 6.0 18.0 1.5 -6.0
run lam2   6.0 18.0 2.0 -6.0
run lam3   6.0 18.0 3.0 -6.0
echo ""; echo "Job finished at $(date)"
