#!/bin/bash
#SBATCH --job-name=r2-multicond
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/r2_multicondition-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/r2_multicondition-%j.log

# R2 -- MULTI-CONDITION MERT training. Half the performances are served from
# the clean embedding bank, half from a bank encoded by pushing acoustically
# DEGRADED waveforms (random spectral tilt + room IR + pink noise) through
# frozen MERT: scripts/precompute_mert_augmented.py, 6615/6615 built, 0 fail.
#
# This is the highest-ceiling track we have. Our synth->real drop is ~45 points
# (B1a 90.0 -> 38.5 room) against CYOLO's 4.3, i.e. a domain-shift failure, and
# multi-condition training is the standard remedy. Unlike R1 (test-time CMN,
# worth at most +1.5) the degradation is baked into the frozen-MERT features and
# cannot be undone by a normalisation layer.
#
# B6 attempted this and finished LAST on room (15.6) because it augmented the
# CBEncoder branch -- 20 points behind MERT before any augmentation -- and used
# reverb only, when the dominant synth->real difference is a static per-band
# gain. R2 fixes both.
#
# Warm-starts from B1a_mert_swap (clean MERT base, 38.5 room), NOT the stronger
# pitch-aux model, so the delta is attributable to multi-condition training
# alone. Adds no parameters, so run_eval_native_mert.py evaluates it unchanged.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/R2_multicondition
mkdir -p "$OUT/runs" "$OUT/params"

# Resume from our own latest; on the FIRST run warm-start from B1a_mert_swap.
# No new parameters are introduced -- the augmentation is a forward-pass
# transform -- so this is a plain fine-tune of the converged base.
#
# Paths MUST be absolute -- this script cd's to $REPO/audio_conditioned_unet
# before invoking python, so a relative checkpoint path resolves against the
# wrong directory and train_model.py dies with FileNotFoundError on a
# checkpoint that does exist (this is what killed jobs 66850715/16/17).
PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    LATEST_CKPT=$(readlink -f "$LATEST_CKPT")
    echo "Resuming from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    BASE=$(find /project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B1a_mert_swap/params \
           -name "best_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$BASE" ] && BASE=$(readlink -f "$BASE")
    if [ -n "$BASE" ] && [ -f "$BASE" ]; then
        echo "Cold start: warm-starting from B1a_mert_swap -> $BASE"
        PARAM_FLAG="--param_path $BASE"
    else
        echo "FATAL: no B1a_mert_swap checkpoint to warm-start from"; exit 1
    fi
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid
export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

# Degraded bank for the TRAIN set only. Validation stays clean on purpose: val
# loss must remain comparable to every other experiment's, otherwise "did
# augmentation help" cannot be read off the training curve at all.
export MERT_AUG_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_aug/train_full"
export MERT_AUG_PROB=0.5
export R2_TRAIN_SET="${TRAIN_SET}"   # lets the entry point verify the map covers it

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== R2: multi-condition MERT (clean + acoustically degraded banks) ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_r2_multicondition.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag R2_multicondition \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
