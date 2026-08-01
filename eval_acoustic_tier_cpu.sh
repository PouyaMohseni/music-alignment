#!/bin/bash
#SBATCH --job-name=eval-acoustic-tier
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_acoustic-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_acoustic-%j.log

# Evaluate a PLAIN-CBEncoder checkpoint on an acoustic-shift tier built by
# scripts/build_acoustic_tier.py. The tier reuses MSMD's sheet images, coords
# and note timing and changes only the audio, so any score delta versus the
# synthetic result is attributable to acoustic-domain shift alone.
#
#   sbatch eval_acoustic_tier_cpu.sh <EXPERIMENT_DIR> <TIER>
#   e.g. sbatch eval_acoustic_tier_cpu.sh B6_impulse_response room
#
# ############################ IMPORTANT ############################
# ONLY for models whose audio encoder is CBEncoder (A0, B2..B6, C2 ...).
# DO NOT use this for B1a / MERT_* checkpoints. extensions/hooks/mert_patch.py
# resolves audio to a PRECOMPUTED embedding file keyed by piece name
# ({emb_root}/{piece}_tempo_{T}.npy) -- those were encoded from the ORIGINAL
# clean audio, so a MERT model run against a degraded tier would silently read
# clean-audio features, the acoustic shift would be invisible, and the run
# would "prove" MERT is perfectly robust. For MERT models first run
# scripts/precompute_mert_acoustic_tier.py on the tier's wavs and point
# MERT_TEST_EMB_ROOT at that output.
# ###################################################################
#
# Read the `ctrl` tier FIRST: it is the same audio content routed through
# CPJKU's real_perf=True path, so it must reproduce the synthetic number. If
# it does not, the harness is wrong and timbre/room mean nothing.

set -uo pipefail
EXP="${1:?usage: sbatch eval_acoustic_tier_cpu.sh <EXPERIMENT_DIR> <TIER>}"
TIER="${2:?usage: sbatch eval_acoustic_tier_cpu.sh <EXPERIMENT_DIR> <TIER>}"
TIER_DIR=/scratch/pmohseni/acoustic_tiers/$TIER

echo "Job started on $(hostname) at $(date)"
echo "experiment=$EXP  tier=$TIER  ($TIER_DIR)"
cd /project/def-ichiro/pmohseni/music-alignment

[ -d "$TIER_DIR/performance" ] || { echo "FATAL: tier not built: $TIER_DIR"; exit 1; }
echo "tier wavs: $(ls $TIER_DIR/performance/*.wav 2>/dev/null | wc -l)"

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

CKPT_DIR=$(find results/cb_ta_ext/$EXP/params -maxdepth 1 -name "*_$EXP" -type d 2>/dev/null | sort | tail -1)
[ -n "$CKPT_DIR" ] || { echo "FATAL: no checkpoint dir for $EXP"; exit 1; }
CKPT="$(readlink -f "$CKPT_DIR/best_model.pt")"
echo "checkpoint: $CKPT"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIG=/project/def-ichiro/pmohseni/music-alignment/configs/msmd_realperf.yaml
REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo ""
echo "=== eval_model.py: $EXP on acoustic tier '$TIER' (real_perf=True) ==="
python eval_model.py \
    --param_path  "$CKPT" \
    --test_dir    "$TIER_DIR" \
    --config      "$CONFIG" \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --eval_onsets \
    --piecewise_stats

echo ""
echo "Job finished at $(date)"
