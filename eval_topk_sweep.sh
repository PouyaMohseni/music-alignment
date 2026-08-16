#!/bin/bash
#SBATCH --job-name=topk-sweep
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/topk_sweep-%j.log
# topk caps how many detections the temporal filter may choose among. At 32 the
# correct candidate can be truncated on crowded frames before the filter ever
# sees it. Map the curve: does it keep rising, or is 128 a fluke of one draw?
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
REC=/scratch/pmohseni/omr/topk; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0 CLUSTER_PX=0
run () { export C2_TOPK=$2 REC_OUT="$REC/$1_room.npz"
    echo ""; echo "##### $1 topk=$2"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]" | grep -iE "^<= |^Average accuracy for Bar|error|Traceback"; }
for K in 16 32 64 128 256 512 2048 100000; do run k$K $K; done
echo ""; echo "Job finished at $(date)"
