#!/bin/bash
#SBATCH --job-name=cand-ir
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=11:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cand_ir-%j.log
# Same dump, with the training audio put through a real room first.
#
# The selector currently learns from synthetic audio where the detector's argmax
# is already 92.7% right, then has to work on room recordings where it is 80%.
# It barely sees the confusions it exists to resolve. Reverberation manufactures
# them from data we already have and leaves the labels untouched.
#
# One room per piece, deterministic from the piece name. Trained on the UNION of
# the clean and reverberant dumps, so the selector sees both regimes rather than
# trading one for the other.
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
export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey IR_SEED=0 IR_PROB=1.0
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
CKPT=$CY/trained_models/cyolo_sb/best_model.pt
OUT=/scratch/pmohseni/omr/cand_ir; mkdir -p "$OUT"

run () { local tag=$1 dir=$2 split=$3
    if [ -f "$OUT/$tag.npz" ]; then echo "##### $tag  already present, skipping"; return; fi
    export DUMP_OUT="$OUT/$tag.npz"
    echo ""; echo "##### $tag  ($split)  IR"
    python extensions/hooks/run_eval_dump.py --param_path "$CKPT" \
        --test_dirs "$DATA/$dir" --split_files "$DATA/split_files/$split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$"; }

run valid_c0 msmd_valid valid_c0_split
for i in 0 1 2 3 4 5; do run train_c$i msmd_train train_c${i}_split; done
echo ""; echo "Job finished at $(date)"
