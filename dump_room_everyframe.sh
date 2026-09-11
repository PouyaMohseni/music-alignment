#!/bin/bash
#SBATCH --job-name=dumpef
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dumpef-%j.log
# Candidate dump for EVERY frame of the real recordings, not only the annotated
# onsets, so the leave-one-piece-out selection can be run on the protocol that
# gives the decoder no onset timing. Without this the honest number and the
# comparable number cannot be the same number.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 DUMP_MAXK=256 DUMP_FEATK=256
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK IR_PATH
O=/scratch/pmohseni/omr/candf256_ef; mkdir -p "$O"
export DUMP_OUT="$O/room.npz"
[ -f "$DUMP_OUT" ] && { echo "present"; exit 0; }
echo "##### every-frame candidate dump, room  $(date)"
python extensions/hooks/run_eval_dump.py \
    --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
    --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
  | grep -E "^<= |\[DUMP\]|\[FEAT\]|rror|Traceback"
ls -l "$DUMP_OUT"
