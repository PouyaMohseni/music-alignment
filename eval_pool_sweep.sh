#!/bin/bash
#SBATCH --job-name=pool-sweep
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/pool_sweep-%j.log
# EVIDENCE POOLING: co-located detections are one hypothesis, not several.
# argmax asks "which anchor is most confident"; pooling asks "which POSITION has
# the most evidence". cluster_px=0 is bit-identical to the current 84.7 decoder.
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
REC=/scratch/pmohseni/omr/pool; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0

run () {  # tag cluster_px topk
    export CLUSTER_PX=$2 C2_TOPK=$3 REC_OUT="$REC/$1_room.npz"
    echo ""; echo "##### $1 cluster_px=$2 topk=$3"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]" \
        | grep -iE "^<= 0.5|^Average accuracy for Bar|error|Traceback"
}
run pool_off   0    32     # control, must be 84.7
run pool4      4    32
run pool8      8    32
run pool16    16    32
run pool32    32    32
# pooling only helps if enough anchors survive to pool -- widen the candidate set
run pool16_k64 16   64
run pool16_k128 16 128
run off_k128    0  128     # isolates topk from pooling
echo ""; echo "Job finished at $(date)"
