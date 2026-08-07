#!/bin/bash
#SBATCH --job-name=dump-conf-r3
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dump_conf_r3-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/dump_conf_r3-%j.log

# Per-frame confidence + tracking-error dump for R3 (MERT + gated belief
# propagation) on a real-audio tier, for the calibration study.
# Mirrors eval_any_cpu.sh exactly -- same tier->embedding-root wiring, which is
# the correctness-critical part: pointing a real-audio run at the synthetic MERT
# bank silently scores synthetic audio.
#
#   sbatch dump_confidence_r3_cpu.sh <room|di-left|rp_synth|synth>

set -uo pipefail
TIER="${1:-room}"
EXP=R3_mert_pitch_belief
echo "Job started on $(hostname) at $(date): tier=$TIER"
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
CKPT=$(readlink -f "$CKPT")
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
mkdir -p $REPO_ROOT/results/calibration
export CONF_DUMP_OUT="$REPO_ROOT/results/calibration/R3_${TIER}_allframes.npz"
echo "MERT_TEST_EMB_ROOT=$MERT_TEST_EMB_ROOT ($(ls $EMB/*.npy 2>/dev/null|wc -l) npy)"

cd $REPO_ROOT/third_party/cpjku_unet/audio_conditioned_unet
python $REPO_ROOT/extensions/hooks/run_eval_conf_dump_r3.py \
    --param_path "$CKPT" \
    --test_dir   "$TEST_DIR" \
    --config     "$CONFIG" \
    $SPLIT \
    --scale_factor 3 --batch_size 1 --seq_len 128 --eval_onsets --piecewise_stats

echo "Job finished at $(date)"
