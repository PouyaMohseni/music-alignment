#!/bin/bash
#SBATCH --job-name=eval-adaptnorm
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_adaptnorm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_adaptnorm-%j.log

# R1 PROBE -- zero-retrain test-time adaptive input normalisation (CMN/CMVN).
#
#   sbatch eval_adaptnorm_cpu.sh <EXPERIMENT> <synth|room|di-left> <ALPHA> [MODE]
#
# ALPHA=0 must reproduce the REAL_AUDIO_SWEEP.md number for that cell exactly;
# it is the built-in control that proves the patch is a no-op when disabled.
# ALPHA=1 fully re-estimates the per-band mean from the test signal itself.
#
# No weights change. The ONLY difference between an ALPHA=0 and an ALPHA=1 run
# of the same checkpoint is the normalisation constants fed to the audio tower,
# so any delta is attributable to that operator alone.
#
# See eval_any_cpu.sh for the tier/embedding-root correctness argument -- it is
# reproduced verbatim below because a stale MERT_TEST_EMB_ROOT does not error,
# it silently scores SYNTHETIC audio and reports it as a real-audio result.

set -uo pipefail
EXP="${1:?usage: sbatch eval_adaptnorm_cpu.sh <EXPERIMENT> <synth|room|di-left> <ALPHA> [MODE]}"
TIER="${2:?}"
ALPHA="${3:?}"
MODE="${4:-mean}"

echo "Job started on $(hostname) at $(date)"
echo "experiment=$EXP tier=$TIER alpha=$ALPHA mode=$MODE"
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
NETCFG="$(dirname "$CKPT")/net_config.json"
[ -f "$NETCFG" ] || { echo "FATAL: no net_config.json beside $CKPT"; exit 1; }
echo "checkpoint: $CKPT"

# Which audio tower this checkpoint was trained with decides whether the MERT
# embedding pipeline has to be patched in. Read it off net_config.json rather
# than pattern-matching the experiment name.
ENC=$(python3 -c "import json;print(json.load(open('$NETCFG')).get('audio_encoder','CBEncoder'))")
echo "audio_encoder: $ENC"
if [ "$ENC" = "MERTProjector" ]; then
    export ADAPTNORM_IS_MERT=1
    N_EMB=$(ls $EMB/*.npy 2>/dev/null | wc -l)
    echo "MERT_TEST_EMB_ROOT=$EMB ($N_EMB npy)"
    [ "$N_EMB" -gt 0 ] || { echo "FATAL: no MERT embeddings under $EMB for tier=$TIER"; exit 1; }
else
    export ADAPTNORM_IS_MERT=0
fi

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MERT_TEST_EMB_ROOT="$EMB"
export MERT_EVAL_TEST_DIR="$TEST_DIR"
export ADAPTNORM_MODE="$MODE"
export ADAPTNORM_ALPHA="$ALPHA"
export ADAPTNORM_VAR_SHRINK="${ADAPTNORM_VAR_SHRINK:-0.5}"

cd $REPO_ROOT/third_party/cpjku_unet/audio_conditioned_unet
echo ""
echo "=== ADAPTNORM $EXP | tier=$TIER | alpha=$ALPHA mode=$MODE ==="
python $REPO_ROOT/extensions/hooks/run_eval_native_adaptive_norm.py \
    --param_path "$CKPT" \
    --test_dir   "$TEST_DIR" \
    --config     "$CONFIG" \
    $SPLIT \
    --scale_factor 3 --batch_size 1 --seq_len 128 --eval_onsets --piecewise_stats

echo ""
echo "Job finished at $(date)"
