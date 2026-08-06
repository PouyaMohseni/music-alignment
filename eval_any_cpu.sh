#!/bin/bash
#SBATCH --job-name=eval-any
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_any-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_any-%j.log

# One CPU evaluator for any checkpoint on any tier, so the sweep does not need
# ~60 near-identical scripts.
#
#   sbatch eval_any_cpu.sh <EXPERIMENT> <synth|room|di-left> <WRAPPER>
#
# WRAPPER is a filename under extensions/hooks/ (or "plain" for stock
# eval_model.py). It carries the architecture patches the checkpoint needs.
#
# ###################### THE CORRECTNESS-CRITICAL PART ######################
# MERT models never encode audio at eval time -- extensions/hooks/mert_patch.py
# resolves each piece to a PRECOMPUTED embedding at
# {MERT_TEST_EMB_ROOT}/{piece}_tempo_{tf}.npy. So the embedding root MUST match
# the tier:
#     synth   -> mert_emb_zenodo/msmd_test        (encoded from synthetic audio)
#     room    -> mert_emb_msmd_rec/room           (encoded from REAL room mics)
#     di-left -> mert_emb_msmd_rec/di-left        (encoded from REAL DI pickup)
# Pointing a real-audio run at the synthetic root does NOT error -- the keys
# exist -- it silently scores synthetic audio and reports it as a real-audio
# result. That is a publication-grade wrong number, so both the root and the
# --test_dir string are derived from $TIER here and never passed in by hand.
# MERT_EVAL_TEST_DIR must equal the --test_dir string verbatim, because
# mert_patch looks its map up by that exact key.
# ##########################################################################

set -uo pipefail
EXP="${1:?usage: sbatch eval_any_cpu.sh <EXPERIMENT> <synth|rp_synth|room|di-left> <WRAPPER|plain>}"
TIER="${2:?}"
WRAPPER="${3:-plain}"

echo "Job started on $(hostname) at $(date)"
echo "experiment=$EXP tier=$TIER wrapper=$WRAPPER"
cd /project/def-ichiro/pmohseni/music-alignment

REPO_ROOT=/project/def-ichiro/pmohseni/music-alignment
REC=$REPO_ROOT/third_party/cpjku_unet/data/msmd/msmd_real_performances

case "$TIER" in
  synth)   TEST_DIR="../data/msmd/msmd_test"
           CONFIG="$REPO_ROOT/third_party/cpjku_unet/audio_conditioned_unet/configs/msmd.yaml"
           SPLIT=""; EMB=/scratch/pmohseni/mert_emb_zenodo/msmd_test ;;
  room|di-left)
           TEST_DIR="$REC"
           CONFIG="$REPO_ROOT/configs/msmd_rec_${TIER}.yaml"
           SPLIT="--split_file $REC/rp_split.yaml"; EMB=/scratch/pmohseni/mert_emb_msmd_rec/$TIER ;;
  rp_synth)
           # The matched synthetic control: SAME 25 pages, SAME score/ (symlink)
           # and SAME GT as room/di-left, but training-soundfont audio. The
           # rp_synth->room delta is pure acoustics, with page difficulty and
           # repeat structure held fixed. Uses the SAME rp_split.yaml because it
           # is literally the same page list.
           TEST_DIR=/scratch/pmohseni/acoustic_tiers/rp_synth
           CONFIG="$REPO_ROOT/configs/msmd_rp_synth.yaml"
           SPLIT="--split_file $REC/rp_split.yaml"; EMB=/scratch/pmohseni/mert_emb_rp_synth ;;
  *) echo "FATAL: unknown tier '$TIER'"; exit 1 ;;
esac
[ -f "$CONFIG" ] || { echo "FATAL: no config $CONFIG"; exit 1; }

SETUP_LOCK=$REPO_ROOT/.cpjku_submodule_setup.flock
( flock -w 120 200
  [ -f third_party/cpjku_unet/audio_conditioned_unet/network.py ] || git submodule update --init third_party/cpjku_unet || true
  git -C third_party/cpjku_unet checkout ismir-2020 ) 200>"$SETUP_LOCK"

CKPT=$(find results/cb_ta_ext/$EXP/params -name best_model.pt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "FATAL: no checkpoint for $EXP"; exit 1; }
CKPT=$(readlink -f "$CKPT")   # absolute: we cd away before invoking python
echo "checkpoint: $CKPT"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DINOV2_TILED_ROOT=/scratch/pmohseni/dinov2_emb_tiled_native
export MERT_TEST_EMB_ROOT="$EMB"
export MERT_EVAL_TEST_DIR="$TEST_DIR"
echo "MERT_TEST_EMB_ROOT=$MERT_TEST_EMB_ROOT ($(ls $EMB/*.npy 2>/dev/null|wc -l) npy)"

if [ "$WRAPPER" = "plain" ]; then
  # Checkpoints carrying _ext_* aux heads need lenient_load, else stock
  # eval_model.py aborts on "Unexpected key(s) in state_dict".
  if python -c "
import sys, torch
sd = torch.load('$CKPT', map_location='cpu')
sys.exit(0 if any(k.startswith('_ext_') for k in sd) else 1)" 2>/dev/null; then
    ENTRY=$REPO_ROOT/extensions/hooks/run_eval_native.py
    echo "checkpoint has _ext_* keys -> lenient_load wrapper"
  else
    ENTRY=eval_model.py
  fi
else
  ENTRY=$REPO_ROOT/extensions/hooks/$WRAPPER
  [ -f "$ENTRY" ] || { echo "FATAL: no wrapper $ENTRY"; exit 1; }
fi

cd $REPO_ROOT/third_party/cpjku_unet/audio_conditioned_unet
echo ""
echo "=== $EXP | tier=$TIER | $(basename $ENTRY) ==="
python "$ENTRY" \
    --param_path "$CKPT" \
    --test_dir   "$TEST_DIR" \
    --config     "$CONFIG" \
    $SPLIT \
    --scale_factor 3 --batch_size 1 --seq_len 128 --eval_onsets --piecewise_stats

echo ""
echo "Job finished at $(date)"
