#!/bin/bash
#SBATCH --job-name=selected
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/selected-%j.log
# THE HONEST NUMBER. The configuration below was chosen on the 19-piece
# reverberant validation split and nowhere else:
#
#   ir_only  blend 0.7   95.4      <- selected
#   ir_union blend 0.7   95.1
#   ...
#   hand prior only      94.3
#
# Room has not been consulted. Whatever this returns is the number, and it is
# reported whether or not it beats the 89.7 that was obtained by picking the
# best of five variants after seeing their room scores.
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
M=/scratch/pmohseni/omr/scorer
REC=/scratch/pmohseni/omr/selected; mkdir -p "$REC"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5
export CLUSTER_PX=0 ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { export REC_OUT="$REC/$1_$2.npz"
    echo ""; echo "##### $1  tier=$2  $(basename $SCORER_PATH) blend=$SCORER_BLEND"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/$2_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |^Average accuracy|rror|Traceback"; }

export SCORER_PATH=$M/ir_only.pt SCORER_BLEND=0.0; run control room     # must be 86.5
export SCORER_PATH=$M/ir_only.pt SCORER_BLEND=0.7; run selected room
export SCORER_PATH=$M/ir_union.pt SCORER_BLEND=0.7; run runnerup room   # validation #2
for T in do rp_synth; do
    export SCORER_PATH=$M/ir_only.pt SCORER_BLEND=0.7; run selected "$T"
done
echo ""; echo "Job finished at $(date)"
