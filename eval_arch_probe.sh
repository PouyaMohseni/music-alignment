#!/bin/bash
#SBATCH --job-name=arch-probe
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/arch_probe-%j.log
# Where do new parameters belong? Four questions, all on the frozen checkpoint.
#
#  A. CEILING. Score every class-0 candidate against ground truth and take the
#     best. That is a perfect re-ranker over these features -- the bound on every
#     post-hoc method left, and the number that says whether a learned scorer is
#     worth building or whether the backbone itself is the wall.
#
#  B. RECURRENCE. encode_samples chunks the performance into 40-frame blocks
#     anchored at frame 0, so the newest block the LSTM ingests ends up to 2 s
#     behind the frame being scored. Re-anchor the blocks at the END: same
#     weights, same chunk size, no new parameters, history now current.
#
#  C. PRESENT WINDOW. The fresh half encodes the last 40 frames. Zero all but
#     the last W and see whether it needs 2 s of audio or only the tail.
#
#  D. WHICH HALF OF z. Zero the LSTM half, then the present half. If zeroing the
#     history costs nothing, recurrent capacity is not the thing to add.
#     (This arm crashed last time: z_enc lives in _modules and cannot be
#     assigned a plain function. Now wrapped in an nn.Module.)
#
# Filtering is EXCLUSION-ONLY -- a whitelist silently ate 17 failed configs once.
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
REC=/scratch/pmohseni/omr/arch; mkdir -p "$REC"

# the shipped best decode: beam-1 temporal filter, no candidate cap
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0
export CLUSTER_PX=0 C2_TOPK=100000
export ORACLE=0 ANCHOR=start WINDOW=0 Z_MASK=none

run () { export REC_OUT="$REC/$1_room.npz" ORACLE_OUT="$REC/$1_cand.npz"
    echo ""; echo "##### $1  ORACLE=$ORACLE ANCHOR=$ANCHOR WINDOW=$WINDOW Z_MASK=$Z_MASK"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$"; }

echo "=== A. candidate ceiling (this arm is also the control: 85.9) ==="
export ORACLE=1; run oracle; export ORACLE=0

echo ""; echo "=== B. history anchoring (start = shipped) ==="
export ANCHOR=end; run anchor_end; export ANCHOR=start

echo ""; echo "=== C. present window length (40 = shipped) ==="
for W in 20 10 5; do export WINDOW=$W; run window$W; done
export WINDOW=0

echo ""; echo "=== D. which half of z steers the detector ==="
for M in hist now; do export Z_MASK=$M; run z_$M; done
export Z_MASK=none

echo ""; echo "Job finished at $(date)"
