#!/bin/bash
#SBATCH --job-name=cand-dump
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=11:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cand_dump-%j.log
# Dump the frozen detector's candidate boxes over the TRAIN and VALID splits so a
# selector can be fit off-test. A perfect selector over these candidates scores
# 99.7 against the shipped argmax's 80.0, so this is where the remaining points
# are. Chunked at 60 pieces so a wall-clock timeout costs one chunk, not the run.
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
OUT=/scratch/pmohseni/omr/cand; mkdir -p "$OUT"

run () { local tag=$1 dir=$2 split=$3
    if [ -f "$OUT/$tag.npz" ]; then echo "##### $tag  already present, skipping"; return; fi
    export DUMP_OUT="$OUT/$tag.npz"
    echo ""; echo "##### $tag  ($split)"
    python extensions/hooks/run_eval_dump.py --param_path "$CKPT" \
        --test_dirs "$DATA/$dir" --split_files "$DATA/split_files/$split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$"; }

run valid_c0 msmd_valid valid_c0_split
for i in 0 1 2 3 4 5; do run train_c$i msmd_train train_c${i}_split; done
echo ""; echo "Job finished at $(date)"
