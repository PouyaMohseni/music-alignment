#!/bin/bash
#SBATCH --job-name=time-prior
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/time_prior-%j.log
# A step is one ONSET, not one frame -- --only_onsets drops non-onset frames from
# the dataset entirely. Those steps are 1 to 64 frames apart (p50=5, p99=19), and
# the prior currently expects the same 6 px of travel with the same 18 px of
# slack across all of them.
#
#   mu = fwd_px * s**mu_pow      sig = sigma_px * s**sig_pow      s = dframes/ref
#
# mu_pow=1 = constant tempo. sig_pow=0.5 = random walk in tempo, 1 = multiplicative.
# ctrl (0,0) is bit-identical to the shipped prior and must return 85.9; if it
# does not, the frame plumbing is wrong and nothing else in this log means
# anything.
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
REC=/scratch/pmohseni/omr/timeprior; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0
export CLUSTER_PX=0 C2_TOPK=100000

run () { export REC_OUT="$REC/$1_room.npz" TIME_MU_POW=$2 TIME_SIG_POW=$3 TIME_REF=$4
    echo ""; echo "##### $1  mu_pow=$2 sig_pow=$3 ref=$4"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$"; }

run ctrl      0    0    5      # must reproduce 85.9 exactly
run mu1_s0    1    0    5      # scale the mean only
run mu1_s05   1    0.5  5      # constant tempo + random-walk slack
run mu1_s1    1    1    5      # constant tempo + multiplicative slack
run mu0_s05   0    0.5  5      # widen only, mean fixed
run mu0_s1    0    1    5
run mu05_s05  0.5  0.5  5      # sub-linear mean, in case tempo is not steady
run mu1_s05_r4  1  0.5  4      # ref sensitivity: is the exponent or the anchor
run mu1_s05_r7  1  0.5  7      # doing the work?
echo ""; echo "Job finished at $(date)"
