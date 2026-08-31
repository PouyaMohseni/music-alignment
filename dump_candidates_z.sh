#!/bin/bash
#SBATCH --job-name=cand-z
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cand_z-%j.log
# Re-dump candidates carrying z, the 128-dim conditioning vector the detector is
# actually steered by. Zeroing z collapses the model to 2.6 pct@0.5s, so it is
# the entire audio side; the selector has been ranking candidates without it,
# on box geometry and a scalar objectness, and so cannot express anything
# audio-dependent at all.
#
# Clean and reverberant, train and validation, so the selector can be fitted and
# selected without room being involved anywhere.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 DUMP_MAXK=256
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
CKPT=$CY/trained_models/cyolo_sb/best_model.pt

run () { local out=$1 dir=$2 split=$3
    [ -f "$out" ] && { echo "##### $(basename $out) present, skipping"; return; }
    export DUMP_OUT="$out"
    echo ""; echo "##### $(basename $out)  ir=${IR_PATH:-none}"
    python extensions/hooks/run_eval_dump.py --param_path "$CKPT" \
        --test_dirs "$DATA/$dir" --split_files "$DATA/split_files/$split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |\[DUMP\]|\[Z\]|\[IR\]|rror"; }

# room first: it is what the offline testbed validates against, and it is quick
unset IR_PATH
O=/scratch/pmohseni/omr/candz; mkdir -p "$O"
run "$O/room.npz" msmd_rp room_split
run "$O/valid.npz" msmd_valid valid_c0_split
for i in 0 1 2 3 4 5; do run "$O/train_c$i.npz" msmd_train train_c${i}_split; done

export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey IR_SEED=0 IR_PROB=1.0
OI=/scratch/pmohseni/omr/candz_ir; mkdir -p "$OI"
run "$OI/valid.npz" msmd_valid valid_c0_split
for i in 0 1 2 3 4 5; do run "$OI/train_c$i.npz" msmd_train train_c${i}_split; done
echo ""; echo "Job finished at $(date)"
