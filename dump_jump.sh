#!/bin/bash
#SBATCH --job-name=dumpjump
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-7
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dumpjump_%a-%A.log
# CODA's jump benchmark, spliced into the audio and re-detected.
#
#   task = 4 * detector + 2 * dataset + gapless
#     detector  0 cyolo_sb   1 cyolo_sb_a   (the shipped and the best)
#     dataset   0 room (16 real recordings)  1 test (94 synthetic, CODA's set)
#     gapless   0 an 8-frame silence before each jump   1 no silence at all
#
# EVERY frame is dumped, not only the annotated onsets: a jump lands in a gap
# where nothing sounds, and a decoder that is only ever stepped at onsets never
# experiences it. Accuracy is still read at onsets, which is the published
# metric; the frames between them are what the decoder has to survive.
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
I=${SLURM_ARRAY_TASK_ID}
DETS=(cyolo_sb cyolo_sb_a)
D=${DETS[$((I / 4))]}
case $(((I % 4) / 2)) in
  0) DS=room; DIR=msmd_rp;   SPLIT=room_split ;;
  1) DS=set1; DIR=msmd_test; SPLIT=test_full_split ;;
esac
case $((I % 2)) in
  0) GAP=8; TAG=gap8 ;;
  1) GAP=0; TAG=gap0 ;;
esac
# flat under omr/: creating a nested tree here trips Lustre's DNE with
# "Object is remote" when the parent lands on another metadata target
O=/scratch/pmohseni/omr/jump_${D}_${DS}_${TAG}
S=/scratch/pmohseni/omr/jumpside_${D}_${DS}_${TAG}
mkdir -p "$O" "$S"
export DUMP_OUT="$O/cand.npz" JUMP_SIDECAR="$O/side.npz"
export JUMP_SIDECAR_DIR="$S"
export JUMP_N=3 JUMP_GAP=$GAP JUMP_SEED=0
[ -f "$DUMP_OUT" ] && [ -f "$JUMP_SIDECAR" ] && { echo "present: $O"; exit 0; }
echo "##### $D  $DS  $TAG  3 jumps/piece  $(date)"
python extensions/hooks/run_eval_dump.py \
    --param_path "$CY/trained_models/$D/best_model.pt" \
    --test_dirs "$DATA/$DIR" --split_files "$DATA/split_files/$SPLIT.yaml" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
  | grep -E "^<= |\[DUMP\]|\[FEAT\]|\[JUMP\]|rror|Traceback" | tail -60
ls -l "$DUMP_OUT" "$JUMP_SIDECAR"
