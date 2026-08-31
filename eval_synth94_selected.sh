#!/bin/bash
#SBATCH --job-name=s94sel2
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/s94sel2-%j.log
# The validation-selected configuration on the canonical 94-piece synthetic test
# set -- the set the published 85.1 (CUNet) and 90.8 (CYOLO-SB+A) refer to. The
# hand decode reaches 91.6 there; this asks what the selector adds.
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
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
export SCORER_PATH=/scratch/pmohseni/omr/scorer/ir_only.pt
R=/scratch/pmohseni/omr/s94sel; mkdir -p "$R"
for B in 0.0 0.7; do
    export SCORER_BLEND=$B REC_OUT="$R/b$B.npz"
    echo ""; echo "##### 94-piece synthetic, blend=$B"
    python extensions/hooks/run_eval_search.py \
        --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
        --test_dirs "$DATA/msmd_test" --split_files "$DATA/split_files/test_full_split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |^Average|rror"
done
echo ""; echo "Job finished at $(date)"
