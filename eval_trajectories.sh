#!/bin/bash
#SBATCH --job-name=traj
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/traj-%j.log
# Record the decoded PATH, not just its error, for the baseline and for ours, on
# all three acoustic renderings of the same 16 performances. pct@0.5s cannot
# distinguish a brief wobble from a confident commitment to the wrong repeat of
# a phrase; the trajectory can, and the three tiers give the same music under
# three different recording conditions.
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
T=/scratch/pmohseni/omr/traj; mkdir -p "$T"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0
export CLUSTER_PX=0 C2_TOPK=100000 TIME_SIG_POW=0 TIME_REF=5
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { export REC_OUT="$T/$1_$2.rec.npz" TRAJ_OUT="$T/$1_$2.traj.npz"
    echo ""; echo "##### tier=$2  arm=$1"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/$2_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$"; }

for TIER in room do rp_synth; do
    export C2_CLASSES='' TIME_MU_POW=0;  run baseline "$TIER"
    export C2_CLASSES='0' TIME_MU_POW=1; run ours     "$TIER"
done
echo ""; echo "Job finished at $(date)"
