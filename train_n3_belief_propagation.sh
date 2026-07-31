#!/bin/bash
#SBATCH --job-name=n3-belief-prop
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/n3_belief_propagation-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/n3_belief_propagation-%j.log

# N3: keep CB_TA's LSTM exactly as-is and ADD a zero-init-gated differentiable
# Bayes filter over the score position, injected as a log-prior on the output
# heatmap. CB_TA predicts every frame's heatmap independently, so nothing in
# the architecture forbids the predicted position from teleporting across the
# page between consecutive frames -- which is precisely the measured failure
# (median onset error 0.000s but mean 1.3-12.4s on the failing pieces: exact
# most of the time, catastrophically wrong in bursts). This adds the missing
# constraint as a proper recursive filter: learned 2D transition kernel
# (general, so it can express a staff-system wrap, not just a forward step)
# plus a learned uniform floor so a wrong commitment stays escapable rather
# than becoming lock-in. Warm-started from B1a it computes EXACTLY B1a at
# step zero (asserted in scripts/smoke_test_temporal_arch.py).

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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/N3_belief_propagation
mkdir -p "$OUT/runs" "$OUT/params"

# Resume from our own latest; on the FIRST run warm-start from B1a's converged
# checkpoint so only the new temporal core trains from scratch.
PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    B1A_CKPT=$(find results/cb_ta_ext/B1a_mert_swap/params -name "best_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
               | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$B1A_CKPT" ]; then
        echo "Cold start: warm-starting from B1a best_model.pt -> $B1A_CKPT"
        PARAM_FLAG="--param_path $B1A_CKPT"
    else
        echo "No B1a checkpoint found, training from scratch"
    fi
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid
export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

echo "=== N3: LSTM kept verbatim + zero-init-gated differentiable Bayes filter ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_n3_belief_propagation.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag N3_belief_propagation \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
