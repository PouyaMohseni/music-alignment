#!/bin/bash
#SBATCH --job-name=set1
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:00:00
#SBATCH --array=0-6
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/set1_%a-%A.log
# CODA's Setting I: the full 94-piece MSMD test split with synthetic audio,
# where the published CYOLO-SB scores .893 and CODA .955 at 0.5 s.
#
#   0 argmax   1 prior   2 scorer            onset protocol, whole split
#   3-6        scorer stepped on EVERY frame, one quarter of the split each
#              (pooled afterwards by every_frame.py)
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK IR_PATH
M=/scratch/pmohseni/omr/scorer
T=/scratch/pmohseni/omr/setting1; mkdir -p "$T"
export C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0 CLUSTER_PX=0
export TIME_SIG_POW=0 TIME_REF=5 ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
I=${SLURM_ARRAY_TASK_ID}
SPLIT=$DATA/split_files/test_full_split.yaml
ONS=--only_onsets
case $I in
  0) A=argmax; export SEARCH_KIND=beam BEAM=1 C2_TOPK=100000 C2_CLASSES='' TIME_MU_POW=0 ;;
  1) A=prior;  export SEARCH_KIND=beam BEAM=1 C2_TOPK=100000 C2_CLASSES='0' TIME_MU_POW=1 ;;
  2) A=scorer; export SEARCH_KIND=scorer C2_TOPK=256 TIME_MU_POW=1 \
                      SCORER_PATH=$M/nbr/nbrp64_s0.pt SCORER_BLEND=0.7 ;;
  *) Q=$((I - 3)); A=every_q$Q; SPLIT=$T/test_full_q$Q.yaml; ONS=
     export SEARCH_KIND=scorer C2_TOPK=256 TIME_MU_POW=1 \
            SCORER_PATH=$M/nbr/nbrp64_s0.pt SCORER_BLEND=0.7 ;;
esac
export TRAJ_OUT="$T/$A.traj.npz" REC_OUT="$T/$A.rec.npz"
echo "##### Setting I  $A  split=$(basename $SPLIT)  ${ONS:-every frame}  $(date)"
python extensions/hooks/run_eval_search.py \
    --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
    --test_dirs "$DATA/msmd_test" --split_files "$SPLIT" $ONS 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
  | grep -E "^<= |rror|Traceback|TRAJ"
echo "##### done $(date)"
