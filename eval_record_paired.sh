#!/bin/bash
#SBATCH --job-name=rec-paired
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/rec_paired-%j.log
# Both arms of the C2 comparison, recorded per frame, in ONE job so the data is
# genuinely paired: same node, same weights, same 16 pieces, same metric code.
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
REC=/scratch/pmohseni/omr/c2_paired
mkdir -p "$REC"

for ARM in baseline c2; do
    echo ""; echo "########## $ARM / room ##########"
    if [ "$ARM" = "c2" ]; then
        export C2_ON=1 C2_LAM=1.0 C2_FWD=6.0 C2_SIGMA=18.0
    else
        export C2_ON=0
    fi
    export REC_OUT="$REC/${ARM}_room.npz"
    python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_record.py \
        --param_path "$CKPT" --test_dirs "$DATA/msmd_rp" \
        --split_files "$DATA/split_files/room_split.yaml" --only_onsets \
        --print_piecewise 2>&1 | grep -v "it/s\]" | tail -80
done

echo ""; echo "Job finished at $(date)"
