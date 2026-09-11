#!/bin/bash
#SBATCH --job-name=h1harn
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/h1harn-%j.log
# The MERT-detector cell through the REAL harness, not the offline rollout.
#
# The offline rollout has matched the harness exactly for three cyolo_sb
# scorers, but the H1 path is new code (h1_eval_patch.py serving MERT inside
# run_eval_search). So the h1 cell's seed-0 scorer runs end to end on room and
# must reproduce its offline number from encoder_ablation.py.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK IR_PATH
H1=/scratch/pmohseni/h1_cyolo_mert/H1_cyolo_sb_mert_mc0.5/params/20260812_151840_H1_cyolo_sb_mert_mc0.5/best_model.pt
export H1_EMB_MAP="$DATA/msmd_rp=/scratch/pmohseni/mert_emb_cyolo/msmd_rp_room"
export SEARCH_KIND=scorer C2_TOPK=256 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0 C2_JUMP=-6.0
export TIME_MU_POW=1 TIME_SIG_POW=0 TIME_REF=5 CLUSTER_PX=0
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0
export SCORER_PATH=/scratch/pmohseni/omr/scorer/abl/h1_s0.pt SCORER_BLEND=0.7
export REC_OUT=/scratch/pmohseni/omr/everyframe/h1_s0_room_onsets.rec.npz
echo "##### H1 detector + h1_s0 scorer, room, onset protocol"
python extensions/hooks/run_eval_search.py --param_path "$H1" \
    --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
    --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
    | grep -E "^<= |\[H1\]|rror|Traceback"
