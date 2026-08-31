#!/bin/bash
#SBATCH --job-name=synth94
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/synth94-%j.log
# The decode has never been run on the CANONICAL 94-piece MSMD synthetic test
# split -- the set our Phase-1 numbers were measured on and the one the
# published 85.1 (CUNet) and 90.8 (CYOLO-SB+A) refer to. Every synthetic number
# we have quoted since is from rp_synth, the 16-piece MSMD-Rec subset, which is
# a different and much smaller set. Close that gap.
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
REC=/scratch/pmohseni/omr/synth94; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0
export CLUSTER_PX=0 C2_TOPK=100000 TIME_SIG_POW=0 TIME_REF=5
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { export REC_OUT="$REC/$1.npz"
    echo ""; echo "##### $1"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_test" --split_files "$DATA/split_files/test_full_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$"; }

export C2_CLASSES='' TIME_MU_POW=0;  run baseline
export C2_CLASSES='0' TIME_MU_POW=0; run decode
export C2_CLASSES='0' TIME_MU_POW=1; run timeaware
echo ""; echo "Job finished at $(date)"
