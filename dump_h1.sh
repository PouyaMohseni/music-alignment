#!/bin/bash
#SBATCH --job-name=dumph1
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dumph1_%a-%A.log
# Candidate dumps from the MERT-audio detector (H1), for the encoder ablation.
#
# Mirrors dump_features.sh -- the dumps every cyolo_sb scorer was trained on --
# with one change: the detector hears a precomputed MERT bank instead of its
# own spectrogram. cyolo_sb's train/valid dumps used real-IR audio (IR_PROB=1);
# an IR cannot be convolved into an embedding, so the real-IR-degraded MERT
# bank stands in for it. Room gets a clean bank of the room recordings, since
# the room is the degradation there.
#
#   0 room (FEATK 256; needs precompute_mert_room_cyolo.sh)  1 valid  2-7 train c0-c5
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 DUMP_MAXK=256
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK IR_PATH
H1=/scratch/pmohseni/h1_cyolo_mert/H1_cyolo_sb_mert_mc0.5/params/20260812_151840_H1_cyolo_sb_mert_mc0.5/best_model.pt
IRB=/scratch/pmohseni/mert_emb_cyolo_ir
O=/scratch/pmohseni/omr/candh1; mkdir -p "$O"
I=${SLURM_ARRAY_TASK_ID}
case $I in
  0) OUT=room;  DIR=msmd_rp;    SPLIT=room_split;     FK=256
     BANK=/scratch/pmohseni/mert_emb_cyolo/msmd_rp_room ;;
  1) OUT=valid; DIR=msmd_valid; SPLIT=valid_c0_split; FK=128; BANK=$IRB/msmd_valid ;;
  *) C=$((I - 2)); OUT=train_c$C; DIR=msmd_train; SPLIT=train_c${C}_split; FK=128
     BANK=$IRB/msmd_train ;;
esac
[ -f "$O/$OUT.npz" ] && { echo "present: $O/$OUT.npz"; exit 0; }
export H1_EMB_MAP="$DATA/$DIR=$BANK" DUMP_FEATK=$FK DUMP_OUT="$O/$OUT.npz"
echo "##### H1 dump $OUT  bank=$BANK  featk=$FK  $(date)"
python extensions/hooks/run_eval_dump.py --param_path "$H1" \
    --test_dirs "$DATA/$DIR" --split_files "$DATA/split_files/$SPLIT.yaml" \
    --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
    | grep -E "^<= |\[DUMP\]|\[FEAT\]|\[H1\]|rror|Traceback"
ls -l "$O/$OUT.npz"
echo "##### done $(date)"
